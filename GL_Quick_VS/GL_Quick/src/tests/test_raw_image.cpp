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
/// @brief Unit tests for the v4 raw arena-image deload datapath — the
///        near-memcpy eviction/reload path.
///
/// @details
/// C1 coverage (pool bulk ops + arena raw primitives): the one-mutex bulk
/// grant/return (`GlobalMemoryManager::acquireBlocks` / `releaseBlocks`) and
/// its equivalence to the per-block loop and grant trigger; the arena
/// round-trip through `rawShape` / `fillLiveBitmap` / `emitRawImage` /
/// `restoreForRawLoad` / `fillRawImage` on a fragmented, multi-block arena with
/// interior dead-vid holes (per-vid byte equality, free-list accounting,
/// invariants); the same-arena reload preserving a `PagedVector`'s and a
/// two-level `PagedHashIndex`'s content WITHOUT any element walk or index
/// rebuild (the raw image carries the pages verbatim); and the new
/// `DeloadStats` raw counters folding into the per-phase out/in totals.
///
/// Arena-level round trips drive the primitives with a private memory manager.
/// Memory-level lifecycle tests also exercise the owning block's release paths
/// and reconstruction of canonical lookup state.

#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "or_implication_processor_fixture.hpp"


#include "../memory.hpp"
#include "../prover.hpp"
#include "../memory_infra/deload_stats.hpp"
#include "../memory_infra/dirty_state.hpp"
#include "../memory_infra/global_memory_manager.hpp"
#include "../memory_infra/lb_arena.hpp"
#include "../memory_infra/lb_deload.hpp"
#include "../memory_infra/paged_hash_index.hpp"
#include "../memory_infra/paged_vector.hpp"
#include "../memory_infra/steward.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

namespace {

    // 4 MiB pool / 256 KiB block = 16 blocks; 8 KiB pages = 32 pages/block.
    // Room for a fragmented source arena and its restored copy side by side.
    const gl::StaticMemoryConfig kRawCfg16{ 1 << 22, 1 << 18 };

    // 1 MiB pool / 4 KiB block / 64 B page = 64 pages/block, 16 vids per
    // directory page: a PagedHashIndex reset past 16 data pages goes two-level.
    const gl::StaticMemoryConfig kRawCfgSmallPage{ 1 << 20, 4096, 64 };

    // Distinct per-vid fill byte, so per-vid byte equality is a real check.
    unsigned char vidByte(int32_t v) {
        return static_cast<unsigned char>(v * 7 + 1);
    }

    // Collect an emitRawImage payload into one contiguous buffer.
    std::vector<char> emitToBuffer(const gl::LbArena& a) {
        std::vector<char> out;
        a.emitRawImage([&out](const char* data, int64_t len) {
            out.insert(out.end(), data, data + static_cast<std::size_t>(len));
        });
        return out;
    }

    // Fill a restored arena's pages sequentially from an in-memory payload —
    // the test's stand-in for a file read straight into pool pages.
    void fillFromBuffer(gl::LbArena& a, const std::vector<char>& buf) {
        std::size_t pos = 0;
        a.fillRawImage([&](char* dst, int64_t len) {
            std::memcpy(dst, buf.data() + pos,
                        static_cast<std::size_t>(len));
            pos += static_cast<std::size_t>(len);
        });
        ASSERT_EQ(pos, buf.size());
    }

    // Minimal patterned statement row (value-initialized -> deterministic
    // padding) for the Memory-level round-trip tests.
    gl::IntEncodedExpr rawExpr(int16_t seed) {
        gl::IntEncodedExpr e{};
        e.nameId = seed;
        e.originalId = static_cast<int16_t>(seed * 3 + 1);
        e.validityId = 1;
        e.arity = 1;
        e.maxIteration = -1;
        return e;
    }

    // Fresh per-test raw-deload directory under .debug (name carries "deload"
    // so the purge-safety assert would hold if ever pointed here).
    std::string freshRawDir(const std::string& name) {
        const std::filesystem::path dir =
            std::filesystem::path(".debug") / ("deload_raw_" + name);
        std::error_code ec;
        std::filesystem::remove_all(dir, ec);
        std::filesystem::create_directories(dir, ec);
        return dir.string();
    }

}

// ---- Pool bulk ops ------------------------------------------------------

TEST(raw_image, pool_bulk_acquire_release_equivalence) {
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    char* buf[8];
    m.acquireBlocks(8, buf);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(8));
    ASSERT_EQ(m.peakBlocksInUse(), static_cast<int64_t>(8));
    // Every granted block is distinct.
    for (int i = 0; i < 8; ++i)
        for (int j = i + 1; j < 8; ++j)
            ASSERT_NE(buf[i], buf[j]);
    m.releaseBlocks(buf, 8);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
    // Recycle-before-carve holds across the bulk path: a fresh single grant
    // hands back a previously-released block (carve cursor did not advance).
    char* one = m.acquireBlock();
    bool recycled = false;
    for (int i = 0; i < 8; ++i) if (one == buf[i]) recycled = true;
    ASSERT_TRUE(recycled);
    m.releaseBlock(one);
    // Bulk and single accounting agree: 8 singles net the same in-use count.
    char* singles[8];
    for (int i = 0; i < 8; ++i) singles[i] = m.acquireBlock();
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(8));
    m.releaseBlocks(singles, 8);
    ASSERT_EQ(m.blocksInUse(), static_cast<int64_t>(0));
}

TEST(raw_image, or_source_state_survives_raw_eviction) {
    const std::string dir = freshRawDir("or_source_state");
    gl::GlobalMemoryManager manager;
    manager.init(gl::StaticMemoryConfig{4 << 20, 1 << 18});
    gl::LbMemory memory(&manager);
    const gl::OrImplicationLeaf leaves[] = {{15, 16}, {7, 8}};
    const int levels[] = {4};
    const int64_t key = gl::packInt32Pair(101, 3);
    ASSERT_TRUE(((memory).orImplications).insert(key, 102, leaves, 2, levels, 1, true));
    ((memory).orImplications).pendingFacts.push_back(gl::packInt32Pair(16, 1));
    const int64_t cacheKey = gl::packInt32Pair(16, 3);
    ((memory).orImplicationMatches).publishCache(cacheKey, gl::OrCanonicalCache{16, 52, 0}, [](auto&&) {});
    const int32_t viewId = ((memory).orImplicationMatches).bindView(gl::OrViewIdentity::sourceLeaf(key, 3, 0), cacheKey, 101);
    ((memory).orImplicationMatches).built = true;
    ((memory).orImplicationMatches).classScopesBuilt = true;
    memory.existenceLookups.algebraSources.postings.insert(cacheKey, key);
    memory.existenceLookups.algebraSources.built = true;
    ((memory).existenceReplays).insert(cacheKey, 1);
    memory.existenceLookups.receipts.ensureCurrent(((memory).existenceReplays),
        [](int64_t packed) { return int64_t(gl::Codec<gl::StatementKey>::decode(packed).validity); });
    const int64_t liveBefore = memory.liveBytes();
    const std::string chain = "or-source-raw-storage";
    const int64_t ordinal = 900101;
    gl::lbdeload::dumpLbMemoryRaw(memory, chain, ordinal, dir);
    memory.manager.releaseAllForRawImage();
    memory.manager.markDeloaded();
    memory.manager.markResident();
    gl::lbdeload::loadLbMemoryRaw(memory, chain, ordinal,
        gl::lbdeload::rawFileName(ordinal), dir);
    const gl::OrImplicationSource source = *((memory).orImplications).find(key);
    ASSERT_EQ(source.canonicalId, 102);
    ASSERT_EQ(((memory).orImplications).leaves[source.leafStart].literalId, 15);
    ASSERT_EQ(((memory).orImplications).leaves[source.leafStart + 1].complementId, 8);
    ASSERT_EQ(((memory).orImplications).levels[source.levelStart], 4);
    ASSERT_EQ(((memory).orImplications).pendingSources[0], key);
    ASSERT_EQ(((memory).orImplications).pendingFacts[0], gl::packInt32Pair(16, 1));
    ASSERT_EQ(memory.liveBytes(), liveBefore);
    ASSERT_EQ(((memory).orImplicationMatches).liveViews, 1);
    ASSERT_EQ(((memory).orImplicationMatches).views[viewId].cacheKey, cacheKey);
    ASSERT_TRUE(((memory).orImplicationMatches).built);
    ASSERT_TRUE(((memory).orImplicationMatches).classScopesBuilt);
    ASSERT_TRUE(memory.existenceLookups.algebraSources.built);
    int sourceHits = 0;
    memory.existenceLookups.algebraSources.postings.forEach(cacheKey, [&](int64_t found) {
        ASSERT_EQ(found, key); ++sourceHits;
    });
    ASSERT_EQ(sourceHits, 1);
    int receiptHits = 0;
    memory.existenceLookups.receipts.forEach(3, [&](int64_t found) {
        ASSERT_EQ(found, cacheKey); ++receiptHits;
    });
    ASSERT_EQ(receiptHits, 1);
    ASSERT_TRUE(((memory).orImplicationMatches).postings().containsKey(gl::orPostingKey(
        gl::OrPostingKind::WatchKey, gl::packInt32Pair(52, 3))));
}

TEST(raw_image, pool_bulk_acquire_fires_grant_trigger_once) {
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    int fired = 0;
    m.armGrantTrigger(5, [&fired]() { ++fired; });
    char* buf[8];
    m.acquireBlocks(8, buf);   // crosses 5 within the batch
    ASSERT_EQ(fired, 1);       // self-disarmed: exactly once
    ASSERT_EQ(m.grantsSinceBarrier(), static_cast<int64_t>(8));
    // Already fired: a further batch does not re-fire.
    char* more[2];
    m.acquireBlocks(2, more);
    ASSERT_EQ(fired, 1);
    m.releaseBlocks(buf, 8);
    m.releaseBlocks(more, 2);
}

// ---- Arena raw round-trip: fragmented, multi-block, interior holes ------

TEST(raw_image, arena_roundtrip_fragmented_multiblock) {
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    gl::LbArena a(&m);
    const int32_t pb = m.pageBytes();

    // 40 pages span two page-blocks (32 + 8); write a per-vid byte pattern.
    const int32_t kPages = 40;
    for (int32_t v = 0; v < kPages; ++v) {
        const int32_t got = a.allocPage();
        ASSERT_EQ(got, v);
        std::memset(a.pageAt(v), vidByte(v), static_cast<std::size_t>(pb));
    }
    // Punch interior holes (never the tail vid 39).
    a.freePage(5);
    a.freePage(17);
    a.freePage(23);
    ASSERT_EQ(a.livePages(), 37);
    ASSERT_EQ(a.pageHighWater(), 40);

    const gl::LbArena::RawShape shape = a.rawShape();
    ASSERT_EQ(shape.vidCount, 40);
    ASSERT_EQ(shape.livePages, 37);
    ASSERT_EQ(shape.byteBumpCursor, static_cast<gl::ArenaOffset>(0));

    const int32_t bmBytes = (shape.vidCount + 7) / 8;   // 5
    std::vector<unsigned char> bm(static_cast<std::size_t>(bmBytes), 0xFF);
    a.fillLiveBitmap(bm.data(), bmBytes);
    // Holes clear, everything else set.
    for (int32_t v = 0; v < 40; ++v) {
        const bool bit = (bm[v >> 3] & (1u << (v & 7))) != 0;
        const bool live = (v != 5 && v != 17 && v != 23);
        ASSERT_EQ(bit, live);
    }

    const std::vector<char> file = emitToBuffer(a);
    ASSERT_EQ(static_cast<int64_t>(file.size()),
              static_cast<int64_t>(37) * pb);

    // Restore into a fresh, empty, resident arena.
    gl::LbArena b(&m);
    b.restoreForRawLoad(shape.vidCount, bm.data(), shape.livePages,
                        shape.byteBumpCursor);
    fillFromBuffer(b, file);

    ASSERT_EQ(b.livePages(), 37);
    ASSERT_EQ(b.pageHighWater(), 40);
    // keptBlocks = ceil(37/32) = 2 -> 64 slots, 37 live, 27 free.
    ASSERT_EQ(b.freePageCount(), 27);
    b.assertInvariants();

    // Per-vid byte equality for every live vid.
    for (int32_t v = 0; v < 40; ++v) {
        const bool live = (bm[v >> 3] & (1u << (v & 7))) != 0;
        if (!live) continue;
        ASSERT_TRUE(std::memcmp(a.pageAt(v), b.pageAt(v),
                                static_cast<std::size_t>(pb)) == 0);
    }
}

// Round trip with a dead TAIL vid (legal now that freed vids park on the
// free-vid LIFO instead of tail-popping) plus interior holes: vid bindings
// and live bytes survive, and post-reload allocation reuses the dead vids in
// the CANONICAL ascending order derived from the live bitmap
// (D-228) before minting any fresh tail vid.
TEST(raw_image, arena_roundtrip_dead_tail_and_vid_reuse_after_reload) {
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    gl::LbArena a(&m);
    const int32_t pb = m.pageBytes();

    const int32_t kPages = 12;
    for (int32_t v = 0; v < kPages; ++v) {
        ASSERT_EQ(a.allocPage(), v);
        std::memset(a.pageAt(v), vidByte(v), static_cast<std::size_t>(pb));
    }
    // Interior holes AND the tail vid.
    a.freePage(3);
    a.freePage(7);
    a.freePage(11);                       // dead TAIL vid
    ASSERT_EQ(a.livePages(), 9);
    ASSERT_EQ(a.pageHighWater(), 12);     // live high water — no tail pop
    a.assertInvariants();

    const gl::LbArena::RawShape shape = a.rawShape();
    ASSERT_EQ(shape.vidCount, 12);
    ASSERT_EQ(shape.livePages, 9);
    const int32_t bmBytes = (shape.vidCount + 7) / 8;
    std::vector<unsigned char> bm(static_cast<std::size_t>(bmBytes), 0xFF);
    a.fillLiveBitmap(bm.data(), bmBytes);
    const std::vector<char> file = emitToBuffer(a);
    ASSERT_EQ(static_cast<int64_t>(file.size()),
              static_cast<int64_t>(9) * pb);

    gl::LbArena b(&m);
    b.restoreForRawLoad(shape.vidCount, bm.data(), shape.livePages,
                        shape.byteBumpCursor);
    fillFromBuffer(b, file);
    ASSERT_EQ(b.livePages(), 9);
    ASSERT_EQ(b.pageHighWater(), 12);
    b.assertInvariants();
    for (int32_t v = 0; v < kPages; ++v) {
        const bool live = (v != 3 && v != 7 && v != 11);
        if (!live) continue;
        ASSERT_TRUE(std::memcmp(a.pageAt(v), b.pageAt(v),
                                static_cast<std::size_t>(pb)) == 0);
    }
    // Canonical reload chain: dead vids pop ASCENDING (3, 7, 11), then fresh.
    ASSERT_EQ(b.allocPage(), 3);
    ASSERT_EQ(b.allocPage(), 7);
    ASSERT_EQ(b.allocPage(), 11);
    ASSERT_EQ(b.allocPage(), 12);         // list dry: fresh tail vid
    ASSERT_EQ(b.pageHighWater(), 13);
    b.assertInvariants();
}

TEST(raw_image, arena_roundtrip_empty_header_only) {
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    gl::LbArena a(&m);   // never allocated
    const gl::LbArena::RawShape shape = a.rawShape();
    ASSERT_EQ(shape.vidCount, 0);
    ASSERT_EQ(shape.livePages, 0);
    ASSERT_EQ(shape.byteBumpCursor, static_cast<gl::ArenaOffset>(0));
    int emitCalls = 0;
    a.emitRawImage([&emitCalls](const char*, int64_t) { ++emitCalls; });
    ASSERT_EQ(emitCalls, 0);

    gl::LbArena b(&m);
    b.restoreForRawLoad(0, nullptr, 0, 0);
    int fillCalls = 0;
    b.fillRawImage([&fillCalls](char*, int64_t) { ++fillCalls; });
    ASSERT_EQ(fillCalls, 0);
    ASSERT_EQ(b.livePages(), 0);
    b.assertInvariants();
}

// ---- Dynamic v4 header: bitmap and chain beyond the old fixed 4 KiB -----

TEST(raw_image, dynamic_header_bitmap_beyond_old_fixed_cap) {
    // The Gauss wall this format change fixes: a big LB's live bitmap alone
    // approaches 4 KiB (a ~1000-block LB at production geometry ≈ 32000 vids
    // = a ~4000-byte bitmap) and cannot share the old FIXED 4 KiB header
    // with the chain and the fixed fields — the deliberate capacity tripwire
    // fired. The dynamic header sizes itself from chainLen + bitmap bytes.
    // 33000 vids -> a 4125-byte bitmap -> an 8192-byte header (two alignment
    // units), impossible before, sufficient by construction now. Small
    // 64-byte pages keep the fixture fast (~2 MiB payload); 64 KiB blocks
    // keep the page-table spill inside PtrDirectory's root cap (33000
    // entries * 8 B = ~258 KiB of table = 4 spill blocks at this geometry).
    const std::string dir = freshRawDir("dynamic_header_big_bitmap");
    gl::GlobalMemoryManager m;
    m.init(gl::StaticMemoryConfig{ 1 << 23, 1 << 16, 64 });
    gl::LbMemory lb(&m);
    const int32_t pb = m.pageBytes();
    const int32_t kPages = 33000;            // > 32768: bitmap > 4096 bytes
    for (int32_t v = 0; v < kPages; ++v) {
        const int32_t got = lb.manager.allocPage();
        ASSERT_EQ(got, v);
        std::memset(lb.manager.pageAt(v), vidByte(v),
                    static_cast<std::size_t>(pb));
    }
    // Interior holes in BOTH halves of the bitmap (never the tail vid), so
    // the chunked fill/restore must preserve holes across chunk boundaries.
    lb.manager.freePage(7);
    lb.manager.freePage(32900);
    ASSERT_EQ(lb.manager.livePages(), kPages - 2);

    const std::string chain = "(dynamic_header_big_bitmap_lb)__";
    const int64_t ordinal = 424242;
    const int64_t payload =
        gl::lbdeload::dumpLbMemoryRaw(lb, chain, ordinal, dir);
    ASSERT_EQ(payload, static_cast<int64_t>(kPages - 2) * pb);
    // Header on disk: prefix (48) + chain (32) + bitmap (4125) = 4205 bytes,
    // rounded up to 8192 — the payload stays 4 KiB-aligned.
    const auto fileSize = std::filesystem::file_size(
        std::filesystem::path(dir) / gl::lbdeload::rawFileName(ordinal));
    ASSERT_EQ(static_cast<int64_t>(fileSize),
              static_cast<int64_t>(8192) + payload);

    // Round trip: release the blocks, reload from the image.
    lb.manager.releaseAllForRawImage();
    lb.manager.markDeloaded();
    lb.manager.markResident();
    gl::lbdeload::loadLbMemoryRaw(lb, chain, ordinal,
                                  gl::lbdeload::rawFileName(ordinal), dir);
    ASSERT_EQ(lb.manager.livePages(), kPages - 2);
    ASSERT_EQ(lb.manager.pageHighWater(), kPages);
    // Holes preserved across the chunked round trip: the reloaded bitmap
    // clears exactly the freed vids (this also pins fillLiveBitmap ==
    // concatenated fillLiveBitmapRange chunks at scale).
    const int32_t bmBytes = (kPages + 7) / 8;
    std::vector<unsigned char> bm(static_cast<std::size_t>(bmBytes), 0xFF);
    lb.manager.fillLiveBitmap(bm.data(), bmBytes);
    for (int32_t v : { 7, 32900 })
        ASSERT_TRUE((bm[v >> 3] & (1u << (v & 7))) == 0);
    // Per-vid byte equality spot checks across the whole vid range,
    // including both sides of the 32768-vid chunk boundary.
    for (int32_t v : { 0, 6, 8, 12345, 32767, 32768, 32899, 32901, 32999 }) {
        const unsigned char* page = reinterpret_cast<const unsigned char*>(
            lb.manager.pageAt(v));
        ASSERT_EQ(page[0], vidByte(v));
        ASSERT_EQ(page[pb - 1], vidByte(v));
    }
}

TEST(raw_image, dynamic_header_long_chain_round_trip) {
    // A chain string LONGER than the whole old fixed header: the dynamic
    // header sizes itself from chainLen, and the loader verifies the chain
    // in bounded 4 KiB chunks (this chain spans two).
    const std::string dir = freshRawDir("dynamic_header_long_chain");
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    gl::LbMemory lb(&m);
    const int32_t pb = m.pageBytes();
    for (int32_t v = 0; v < 5; ++v) {
        ASSERT_EQ(lb.manager.allocPage(), v);
        std::memset(lb.manager.pageAt(v), vidByte(v),
                    static_cast<std::size_t>(pb));
    }
    std::string chain = "(long_chain_";
    chain.append(6000, 'k');                 // > 4096: spans two chunks
    chain += ")__";
    const int64_t ordinal = 424243;
    const int64_t payload =
        gl::lbdeload::dumpLbMemoryRaw(lb, chain, ordinal, dir);
    ASSERT_EQ(payload, static_cast<int64_t>(5) * pb);
    // prefix (48) + chain (6015) + bitmap (1) = 6064 -> 8192.
    const auto fileSize = std::filesystem::file_size(
        std::filesystem::path(dir) / gl::lbdeload::rawFileName(ordinal));
    ASSERT_EQ(static_cast<int64_t>(fileSize),
              static_cast<int64_t>(8192) + payload);

    lb.manager.releaseAllForRawImage();
    lb.manager.markDeloaded();
    lb.manager.markResident();
    gl::lbdeload::loadLbMemoryRaw(lb, chain, ordinal,
                                  gl::lbdeload::rawFileName(ordinal), dir);
    ASSERT_EQ(lb.manager.livePages(), 5);
    for (int32_t v = 0; v < 5; ++v) {
        const unsigned char* page = reinterpret_cast<const unsigned char*>(
            lb.manager.pageAt(v));
        ASSERT_EQ(page[0], vidByte(v));
        ASSERT_EQ(page[pb - 1], vidByte(v));
    }
}

// ---- Same-arena reload: containers survive with NO element walk / rebuild

TEST(raw_image, same_arena_reload_paged_vector_intact) {
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    gl::LbArena a(&m);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> pv(&a, &d);
    // > 2048 elements forces multiple data pages + a directory page (default
    // 8 KiB pages hold 2048 int32 each), so the directory page round-trips too.
    const int32_t kN = 5000;
    for (int32_t i = 0; i < kN; ++i) pv.push_back(i * 7 + 1);
    ASSERT_EQ(pv.size(), kN);

    const gl::LbArena::RawShape shape = a.rawShape();
    const int32_t bmBytes = (shape.vidCount + 7) / 8;
    std::vector<unsigned char> bm(static_cast<std::size_t>(bmBytes), 0);
    a.fillLiveBitmap(bm.data(), bmBytes);
    const std::vector<char> file = emitToBuffer(a);

    // Deload the arena WITHOUT touching pv's bookkeeping (the raw-eviction
    // contract: no container.release() walk — rootVid_/size_/numPages_ survive).
    a.releaseAll();
    a.markDeloaded();

    // Raw reload: bind the same vids to fresh dense pages, refill the bytes.
    a.markResident();
    a.restoreForRawLoad(shape.vidCount, bm.data(), shape.livePages,
                        shape.byteBumpCursor);
    fillFromBuffer(a, file);

    // pv resolves against the restored pages — no rebuild, values intact.
    ASSERT_EQ(pv.size(), kN);
    for (int32_t i = 0; i < kN; ++i)
        ASSERT_EQ(pv[i], i * 7 + 1);
}

TEST(raw_image, same_arena_reload_two_level_hash_index_no_rebuild) {
    gl::deloadStats().reset();
    gl::GlobalMemoryManager m;
    m.init(kRawCfgSmallPage);
    gl::LbArena a(&m);
    gl::PagedHashIndex idx(&a);
    // 64 B pages -> 16 slots/page, dirCap 16; 512 slots = 32 data pages > 16,
    // so the directory is two-level (L2 root -> L1 pages -> data pages).
    const int32_t kCap = 512;
    idx.reset(kCap);
    for (int32_t i = 0; i < kCap; ++i) idx.set(i, i * 3 + 2);

    const int64_t rebuildsBefore =
        gl::deloadStats().indexRebuildCount.load();

    const gl::LbArena::RawShape shape = a.rawShape();
    const int32_t bmBytes = (shape.vidCount + 7) / 8;
    std::vector<unsigned char> bm(static_cast<std::size_t>(bmBytes), 0);
    a.fillLiveBitmap(bm.data(), bmBytes);
    const std::vector<char> file = emitToBuffer(a);

    a.releaseAll();
    a.markDeloaded();
    a.markResident();
    a.restoreForRawLoad(shape.vidCount, bm.data(), shape.livePages,
                        shape.byteBumpCursor);
    fillFromBuffer(a, file);

    // The two-level directory AND every slot survived byte-for-byte.
    ASSERT_EQ(idx.capacity(), kCap);
    for (int32_t i = 0; i < kCap; ++i)
        ASSERT_EQ(idx.at(i), i * 3 + 2);
    // The raw path rebuilt NO throw-away index (the whole point).
    ASSERT_EQ(gl::deloadStats().indexRebuildCount.load(), rebuildsBefore);
}

// ---- DeloadStats raw counters ------------------------------------------

TEST(raw_image, deload_stats_raw_counters_accumulate) {
    gl::DeloadStats& s = gl::deloadStats();
    s.reset();
    s.recordRawDump(1000, 500);
    s.recordRawDump(2000, 700);
    s.recordRawLoad(4096, 300);
    ASSERT_EQ(s.rawDumpBytes.load(), static_cast<int64_t>(3000));
    ASSERT_EQ(s.rawDumpNs.load(), static_cast<int64_t>(1200));
    ASSERT_EQ(s.rawDumpCount.load(), static_cast<int64_t>(2));
    ASSERT_EQ(s.rawLoadBytes.load(), static_cast<int64_t>(4096));
    ASSERT_EQ(s.rawLoadCount.load(), static_cast<int64_t>(1));
    // A raw load rebuilds no index; only recordIndexRebuild bumps the count.
    ASSERT_EQ(s.indexRebuildCount.load(), static_cast<int64_t>(0));
    s.recordIndexRebuild(9, 40);
    ASSERT_EQ(s.indexRebuildCount.load(), static_cast<int64_t>(1));
    s.reset();
    ASSERT_EQ(s.rawDumpBytes.load(), static_cast<int64_t>(0));
    ASSERT_EQ(s.indexRebuildCount.load(), static_cast<int64_t>(0));
}

TEST(raw_image, deload_stats_phase_window_folds_raw_into_totals) {
    gl::DeloadStats& s = gl::deloadStats();
    s.reset();
    s.beginPhaseWindow(2);
    // Both streams move during the window: the out/in totals fold them.
    s.recordRawDump(4096, 100);
    s.recordV3Dump(1024, 50);
    s.recordRawLoad(8192, 200);
    s.recordV3Load(512, 30);
    const gl::PhaseDelta d = s.endPhaseWindow();
    ASSERT_EQ(d.phase, 2);
    ASSERT_EQ(d.outBytes, static_cast<int64_t>(4096 + 1024));
    ASSERT_EQ(d.outNs, static_cast<int64_t>(100 + 50));
    ASSERT_EQ(d.inBytes, static_cast<int64_t>(8192 + 512));
    ASSERT_EQ(d.inNs, static_cast<int64_t>(200 + 30));
    s.reset();
}

// ---- C2: Memory-level v4 dump / release / reload dispatch ---------------

TEST(raw_image, memory_raw_roundtrip_preserves_all_containers_no_rebuild) {
    gl::deloadStats().reset();
    const std::string dir = freshRawDir("mem_roundtrip");
    gl::Memory m;
    m.setExprKey("(raw_image_mem_roundtrip_lb)");
    const int16_t kStmts = 40;
    for (int16_t i = 0; i < kStmts; ++i)
        ((m).intEncodedStatements).push_back(rawExpr(i));
    // Populate a real cold-map (ColdHashSet with a PagedHashIndex bucket array)
    // so "no rebuild" is a meaningful claim (the buckets must round-trip).
    const int32_t kKeys = 50;
    for (int32_t i = 0; i < kKeys; ++i)
        ((m).intLocalEncodedStatementsSet).mint(7 * i + 3);
    ASSERT_TRUE(((m).intLocalEncodedStatementsSet).contains(7 * 10 + 3));

    const gl::NameId releaseScope = ((m).nameMap).encode("main");
    const gl::NameId releaseOriginal = ((m).nameMap).encode("(or90[a,b])");
    const int64_t releasePair = gl::packInt32Pair(releaseOriginal, releaseOriginal);


    auto& canonical=((m).lbMemory).canonicalUses;
    canonical.statements.add(releasePair,releaseScope,gl::StrSpan("a"));
    canonical.pendingCarriers.insert(releasePair,91);
    canonical.pendingIdentity.upsert(91,releasePair);
    canonical.pendingBuilt=true;
    canonical.statementMatches.forms.upsert(releasePair,17);
    canonical.statementMatches.rebuildCount=9;
    gl::EquivalenceClass complete; complete.memberIds={2,3};
    ((m).canonicalEquivalenceClassesMap).assignRun(releaseScope,{complete});

    m.deloadStaticContainersRaw(dir);           // v4 dump + release
    ASSERT_FALSE(((m).lbMemory).manager.resident());
    ASSERT_EQ(((m).deloadKind), gl::Memory::DeloadKind::Raw);
    ASSERT_EQ(static_cast<int>(((m).deloadFiles).size()), 1);

    const int64_t rebuildsBefore =
        gl::deloadStats().indexRebuildCount.load();
    m.ensureLoaded(dir);                         // dispatch -> raw loader
    ASSERT_TRUE(((m).lbMemory).manager.resident());
    // The raw reload rebuilt NO throw-away index — the whole point.
    ASSERT_EQ(gl::deloadStats().indexRebuildCount.load(), rebuildsBefore);
    ASSERT_TRUE(gl::deloadStats().rawLoadCount.load() > 0);

    // Statements intact.
    int hits=0;
    canonical.statements.forEach(releaseScope,gl::StrSpan("a"),[&](int64_t key){ASSERT_EQ(key,releasePair);++hits;});
    ASSERT_EQ(hits,1);
    ASSERT_EQ(*canonical.pendingIdentity.find(91),releasePair);
    ASSERT_TRUE(canonical.pendingBuilt);
    ASSERT_EQ(*canonical.statementMatches.forms.find(releasePair),17);
    ASSERT_EQ(canonical.statementMatches.rebuildCount,9);
    ASSERT_EQ(((m).canonicalEquivalenceClassesMap).recordAt(((m).canonicalEquivalenceClassesMap).lookup(releaseScope),0).memberIds.size(),size_t(2));
    ASSERT_EQ(((m).intEncodedStatements).size(), kStmts);
    for (int16_t i = 0; i < kStmts; ++i) {
        ASSERT_EQ(((m).intEncodedStatements)[i].nameId, rawExpr(i).nameId);
        ASSERT_EQ(((m).intEncodedStatements)[i].originalId,
                  rawExpr(i).originalId);
    }
    // Cold-map lookups still hit — buckets_ survived byte-identical.
    for (int32_t i = 0; i < kKeys; ++i)
        ASSERT_TRUE(((m).intLocalEncodedStatementsSet).contains(7 * i + 3));
    ASSERT_FALSE(((m).intLocalEncodedStatementsSet).contains(999999));
}

TEST(raw_image, memory_raw_dump_writes_single_named_file) {
    gl::deloadStats().reset();
    const std::string dir = freshRawDir("single_file");
    gl::Memory m;
    m.setExprKey("(raw_image_single_file_lb)");
    for (int16_t i = 0; i < 12; ++i)
        ((m).intEncodedStatements).push_back(rawExpr(i));

    m.dumpStaticContainersRaw(dir);             // dump only (stays resident)
    ASSERT_TRUE(((m).lbMemory).manager.resident());
    ASSERT_EQ(((m).deloadKind), gl::Memory::DeloadKind::Raw);
    ASSERT_EQ(static_cast<int>(((m).deloadFiles).size()), 1);
    const std::string expected = gl::lbdeload::rawFileName(((m).deloadOrdinal));
    ASSERT_EQ(((m).deloadFiles).front(), expected);
    ASSERT_TRUE(std::filesystem::exists(
        std::filesystem::path(dir) / expected));
    ASSERT_EQ(gl::deloadStats().rawDumpCount.load(),
              static_cast<int64_t>(1));
    ASSERT_TRUE(m.lastRawImageBytes() > 0);
}

TEST(raw_image, memory_raw_skip_clean_on_resident_redump) {
    gl::deloadStats().reset();
    const std::string dir = freshRawDir("skip_clean");
    gl::Memory m;
    m.setExprKey("(raw_image_skip_clean_lb)");
    for (int16_t i = 0; i < 8; ++i)
        ((m).intEncodedStatements).push_back(rawExpr(i));

    m.dumpStaticContainersRaw(dir);
    ASSERT_EQ(gl::deloadStats().rawDumpCount.load(),
              static_cast<int64_t>(1));
    ASSERT_EQ(gl::deloadStats().evictSkippedClean.load(),
              static_cast<int64_t>(0));
    // Clean + a raw base on disk: the second dump skips the write.
    m.dumpStaticContainersRaw(dir);
    ASSERT_EQ(gl::deloadStats().rawDumpCount.load(),
              static_cast<int64_t>(1));
    ASSERT_EQ(gl::deloadStats().evictSkippedClean.load(),
              static_cast<int64_t>(1));
}

TEST(raw_image, memory_raw_cold_metadata_readable_while_deloaded) {
    const std::string dir = freshRawDir("cold_meta");
    gl::Memory m;
    m.setExprKey("(raw_image_cold_meta_lb)");
    const int32_t kStmts = 25;
    for (int16_t i = 0; i < static_cast<int16_t>(kStmts); ++i)
        ((m).intEncodedStatements).push_back(rawExpr(i));

    m.deloadStaticContainersRaw(dir);           // dump + release
    ASSERT_FALSE(((m).lbMemory).manager.resident());
    // The count reads deloadedCounts while cold — no reload triggered.
    ASSERT_EQ(m.intEncodedStatementsCount(), kStmts);
}

TEST(raw_image, canonical_cache_only_mutation_requires_a_fresh_raw_snapshot) {
    gl::deloadStats().reset();
    const auto directory=freshRawDir("canonical_cache_mutation");
    gl::Memory memory;
    memory.setExprKey("(canonical_cache_mutation)");
    auto& uses=((memory).lbMemory).canonicalUses;
    const auto root=gl::NameMap::MAIN_ID;
    uses.statements.add(11,root,gl::StrSpan("a"));
    ASSERT_TRUE(uses.statements.hasRawChanges());
    uses.statements.markRawClean();
    ASSERT_FALSE(uses.statements.hasRawChanges());
    uses.statementMatches.forms.upsert(21,31);
    ASSERT_TRUE(uses.statementMatches.hasRawChanges());
    uses.statementMatches.markRawClean();
    ASSERT_FALSE(uses.statementMatches.hasRawChanges());
    uses.markRawClean();
    ASSERT_FALSE(uses.hasRawChanges());
    memory.dumpStaticContainersRaw(directory);
    const auto dumps=gl::deloadStats().fullDumpCount.load();
    uses.statements.add(12,root,gl::StrSpan("b"));
    uses.statementMatches.forms.upsert(21,32);
    ASSERT_EQ(((memory).lbMemory).dirty,gl::DirtyState::Clean);
    ASSERT_TRUE(uses.hasRawChanges());
    memory.deloadStaticContainersRaw(directory);
    ASSERT_EQ(gl::deloadStats().fullDumpCount.load(),dumps+1);
    memory.ensureLoaded(directory);
    int hits=0;
    uses.statements.forEach(root,gl::StrSpan("b"),[&](int64_t key){ASSERT_EQ(key,12);++hits;});
    ASSERT_EQ(hits,1);
    ASSERT_EQ(*uses.statementMatches.forms.find(21),32);
    ASSERT_FALSE(uses.hasRawChanges());
    memory.dumpStaticContainersRaw(directory);
    ASSERT_EQ(gl::deloadStats().fullDumpCount.load(),dumps+1);
}

TEST(raw_image, v3_image_still_loads_through_dispatch_and_rebuilds) {
    gl::deloadStats().reset();
    const std::string dir = freshRawDir("v3_dispatch");
    gl::Memory m;
    m.setExprKey("(raw_image_v3_dispatch_lb)");
    for (int16_t i = 0; i < 20; ++i)
        ((m).intEncodedStatements).push_back(rawExpr(i));

    m.deloadStaticContainers(dir);              // v3 canonical dump + release
    ASSERT_EQ(((m).deloadKind), gl::Memory::DeloadKind::Canonical);
    ASSERT_FALSE(((m).lbMemory).manager.resident());

    const int64_t rebuildsBefore =
        gl::deloadStats().indexRebuildCount.load();
    m.ensureLoaded(dir);                         // dispatch -> v3 loader
    ASSERT_TRUE(((m).lbMemory).manager.resident());
    // The v3 path DOES rebuild the throw-away indexes (the contrast with raw).
    ASSERT_TRUE(gl::deloadStats().indexRebuildCount.load() > rebuildsBefore);
    ASSERT_TRUE(gl::deloadStats().v3LoadCount.load() > 0);
    ASSERT_EQ(((m).intEncodedStatements).size(), static_cast<int32_t>(20));
}

TEST(raw_image, teardown_of_raw_deloaded_memory_does_not_assert) {
    const std::string dir = freshRawDir("teardown");
    {
        gl::Memory m;
        m.setExprKey("(raw_image_teardown_lb)");
        for (int16_t i = 0; i < 30; ++i)
            ((m).intEncodedStatements).push_back(rawExpr(i));
        for (int32_t i = 0; i < 40; ++i)
            ((m).intLocalEncodedStatementsSet).mint(11 * i + 5);
        m.deloadStaticContainersRaw(dir);
        ASSERT_FALSE(((m).lbMemory).manager.resident());
        // m destructs HERE on a deloaded arena: ~PagedVector / ~PagedHashIndex
        // must take their teardown residency branch and free nothing (the
        // blocks are already back in the pool) instead of asserting in
        // freePage. The inverse property — a LIVE clear() on a cold arena
        // still dies on freePage's residency assert — is untestable here: the
        // harness has no assert-death pattern (assert() aborts the process),
        // so it is guaranteed by the branch living ONLY in the destructors.
    }
    ASSERT_TRUE(true);   // reached the end -> teardown did not abort
}

// ---- C3: eviction paths wired to raw; discharge stays v3 ----------------

TEST(raw_image, steward_eviction_writes_raw_image_and_reloads) {
    gl::deloadStats().reset();
    const std::string dir = freshRawDir("steward_evict");
    gl::Memory victim;
    victim.setExprKey("(raw_image_steward_evict_victim)");
    ((victim).isActive) = true;                     // no default initializer
    for (int16_t i = 0; i < 30; ++i)
        ((victim).intEncodedStatements).push_back(rawExpr(i));
    gl::Memory keep;
    keep.setExprKey("(raw_image_steward_evict_keep)");
    ((keep).isActive) = true;
    ((keep).intEncodedStatements).push_back(rawExpr(0));

    // order = {victim, keep}; cursor 2, workers 1 -> kept window [1, 5), so
    // the victim at index 0 is OUTSIDE the kept range and the two-tier
    // Belady scan (backward from keepLo - 1 = 0) picks it; keep at index 1
    // is protected. Driven synchronously (steward thread not started) so no
    // background pass races.
    std::vector<gl::Memory*> order = { &victim, &keep };
    std::atomic<std::size_t> cursor{ 2 };
    gl::MemorySteward s;
    s.beginPhaseWindow(1, &cursor, &order, /*workers=*/1u, dir);
    const bool evicted = s.evictOneForReload(dir);
    s.endPhaseWindow();

    ASSERT_TRUE(evicted);
    ASSERT_FALSE(((victim).lbMemory).manager.resident());
    ASSERT_EQ(((victim).deloadKind), gl::Memory::DeloadKind::Raw);
    ASSERT_EQ(static_cast<int>(((victim).deloadFiles).size()), 1);
    ASSERT_EQ(((victim).deloadFiles).front(),
              gl::lbdeload::rawFileName(((victim).deloadOrdinal)));
    ASSERT_TRUE(std::filesystem::exists(
        std::filesystem::path(dir)
        / gl::lbdeload::rawFileName(((victim).deloadOrdinal))));
    ASSERT_TRUE(gl::deloadStats().rawDumpCount.load() > 0);
    ASSERT_EQ(((victim).stewardClaim).load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    ASSERT_TRUE(((keep).lbMemory).manager.resident());   // inside the window

    // Reload the raw-evicted victim: content intact, zero index rebuild.
    const int64_t rb = gl::deloadStats().indexRebuildCount.load();
    victim.ensureLoaded(dir);
    ASSERT_TRUE(((victim).lbMemory).manager.resident());
    ASSERT_EQ(gl::deloadStats().indexRebuildCount.load(), rb);
    ASSERT_EQ(((victim).intEncodedStatements).size(), static_cast<int32_t>(30));
    ASSERT_EQ(((victim).intEncodedStatements)[29].nameId,
              static_cast<int16_t>(29));
}

TEST(raw_image, worker_claim_and_load_round_trip_on_raw_image) {
    gl::deloadStats().reset();
    const std::string dir = freshRawDir("worker_reload");
    gl::Memory lb;
    lb.setExprKey("(raw_image_worker_reload_lb)");
    ((lb).isActive) = true;
    for (int16_t i = 0; i < 22; ++i)
        ((lb).intEncodedStatements).push_back(rawExpr(i));

    // Raw-deload and mark Dumped (the state a steward eviction leaves).
    lb.deloadStaticContainersRaw(dir);
    ((lb).stewardClaim).store(
        static_cast<uint8_t>(gl::Memory::StewardClaim::Dumped));
    ASSERT_FALSE(((lb).lbMemory).manager.resident());

    // The worker handshake self-loads the cold raw image and publishes
    // WorkerOwned. No window open -> no inline evict; a plain self-load.
    gl::MemorySteward s;
    const int64_t rb = gl::deloadStats().indexRebuildCount.load();
    s.claimAndLoadForWork(lb, /*phase=*/2, dir);
    ASSERT_TRUE(((lb).lbMemory).manager.resident());
    ASSERT_EQ(((lb).stewardClaim).load(),
              static_cast<uint8_t>(gl::Memory::StewardClaim::WorkerOwned));
    ASSERT_EQ(gl::deloadStats().indexRebuildCount.load(), rb);   // raw: no rebuild
    ASSERT_TRUE(gl::deloadStats().rawLoadCount.load() > 0);
    ASSERT_EQ(((lb).intEncodedStatements).size(), static_cast<int32_t>(22));
    for (int16_t i = 0; i < 22; ++i)
        ASSERT_EQ(((lb).intEncodedStatements)[i].nameId, i);
}

TEST(raw_image, discharge_drain_still_writes_canonical_v3) {
    const std::string dir = freshRawDir("discharge_v3");
    gl::Memory lb;
    lb.setExprKey("(raw_image_discharge_v3_lb)");
    for (int16_t i = 0; i < 18; ++i)
        ((lb).intEncodedStatements).push_back(rawExpr(i));

    // The discharge drain path: installDischargeWork -> deloadStaticContainers.
    gl::MemorySteward s;
    s.start();
    s.installDischargeWork({ &lb }, dir);
    s.wake();
    s.quiesce();
    s.stop();

    ASSERT_FALSE(((lb).lbMemory).manager.resident());
    // Discharge writes the CANONICAL v3 format, NOT the raw image.
    ASSERT_EQ(((lb).deloadKind), gl::Memory::DeloadKind::Canonical);
    ASSERT_FALSE(((lb).deloadFiles).empty());
    ASSERT_TRUE(((lb).deloadFiles).front().find("_of_") != std::string::npos);
    ASSERT_TRUE(((lb).deloadFiles).front().find("_raw") == std::string::npos);
    // And it reloads through the v3 dispatch.
    lb.ensureLoaded(dir);
    ASSERT_EQ(((lb).intEncodedStatements).size(), static_cast<int32_t>(18));
}

TEST(raw_image, mixed_raw_evict_then_discharge_canonical) {
    const std::string dir = freshRawDir("mixed");
    // Discharge compaction scratch on an INDEPENDENT pool (never-deloaded),
    // as production uses the LB-body pool, not the deloadable one.
    gl::GlobalMemoryManager gScr;
    gScr.init(kRawCfg16);
    gl::LbArena scratch(&gScr);

    gl::Memory lb;
    lb.setExprKey("(raw_image_mixed_lb)");
    ((lb).isActive) = true;
    for (int16_t i = 0; i < 35; ++i)
        ((lb).intEncodedStatements).push_back(rawExpr(i));

    // 1) Raw-evict, then raw-reload.
    lb.deloadStaticContainersRaw(dir);
    ASSERT_EQ(((lb).deloadKind), gl::Memory::DeloadKind::Raw);
    lb.ensureLoaded(dir);
    ASSERT_TRUE(((lb).lbMemory).manager.resident());
    ASSERT_EQ(((lb).intEncodedStatements).size(), static_cast<int32_t>(35));

    // 2) The LB becomes inactive and is discharged -> canonical v3 next dump.
    lb.dischargeStatementContent(scratch);
    lb.deloadStaticContainers(dir);
    ASSERT_EQ(((lb).deloadKind), gl::Memory::DeloadKind::Canonical);
    ASSERT_TRUE(((lb).deloadFiles).front().find("_raw") == std::string::npos);
    ASSERT_FALSE(((lb).lbMemory).manager.resident());

    // 3) The chapter export reads the discharged v3 image (dispatch -> v3).
    lb.ensureLoadedForRead(dir);
    ASSERT_TRUE(((lb).lbMemory).manager.resident());
}

// ---- C5: read-path tuning + policy interface surface --------------------

TEST(raw_image, multi_block_raw_reload_correctness) {
    gl::deloadStats().reset();
    const std::string dir = freshRawDir("multi_block");
    gl::Memory m;
    m.setExprKey("(raw_image_multi_block_lb)");
    // Enough statements that the arena spans MORE THAN ONE 256 KiB block, so
    // the reload's fillRawImage emits several block-region reads (the tuning).
    const int32_t kN = 10000;
    for (int32_t i = 0; i < kN; ++i)
        ((m).intEncodedStatements).push_back(rawExpr(static_cast<int16_t>(i)));
    ASSERT_TRUE(((m).lbMemory).manager.blocksHeld() > 1);   // genuinely multi-block

    m.deloadStaticContainersRaw(dir);
    ASSERT_FALSE(((m).lbMemory).manager.resident());
    const int64_t rb = gl::deloadStats().indexRebuildCount.load();
    m.ensureLoaded(dir);
    ASSERT_TRUE(((m).lbMemory).manager.resident());
    ASSERT_EQ(gl::deloadStats().indexRebuildCount.load(), rb);   // no rebuild
    ASSERT_EQ(((m).intEncodedStatements).size(), kN);
    // Byte-exact across the whole multi-block span.
    for (int32_t i = 0; i < kN; ++i)
        ASSERT_EQ(((m).intEncodedStatements)[i].nameId, static_cast<int16_t>(i));
}

TEST(raw_image, last_raw_image_bytes_accessor_reflects_dump) {
    const std::string dir = freshRawDir("last_bytes");
    gl::Memory m;
    m.setExprKey("(raw_image_last_bytes_lb)");
    ASSERT_EQ(m.lastRawImageBytes(), static_cast<int64_t>(0));   // pre-dump
    for (int16_t i = 0; i < 100; ++i)
        ((m).intEncodedStatements).push_back(rawExpr(i));

    m.dumpStaticContainersRaw(dir);                 // dump only (stays resident)
    const int64_t bytes = m.lastRawImageBytes();
    ASSERT_TRUE(bytes > 0);
    // Equals livePages * pageBytes + the byte-bump span (0 in production).
    const gl::LbArena::RawShape shape = ((m).lbMemory).manager.rawShape();
    ASSERT_EQ(bytes,
              static_cast<int64_t>(shape.livePages)
                  * static_cast<int64_t>(shape.pageBytes)
              + static_cast<int64_t>(shape.byteBumpCursor));

    // Readable while deloaded (a plain shell scalar — no residency assert).
    m.releaseStaticBlocksRaw();
    ASSERT_FALSE(((m).lbMemory).manager.resident());
    ASSERT_EQ(m.lastRawImageBytes(), bytes);
}

TEST(raw_image, raw_load_peak_latency_tracked) {
    gl::DeloadStats& s = gl::deloadStats();
    s.reset();
    ASSERT_EQ(s.rawLoadPeakNs.load(), static_cast<int64_t>(0));
    s.recordRawLoad(100, 500);
    ASSERT_EQ(s.rawLoadPeakNs.load(), static_cast<int64_t>(500));
    s.recordRawLoad(200, 300);   // faster ns -> peak unchanged
    ASSERT_EQ(s.rawLoadPeakNs.load(), static_cast<int64_t>(500));
    s.recordRawLoad(50, 900);    // new worst case
    ASSERT_EQ(s.rawLoadPeakNs.load(), static_cast<int64_t>(900));
    ASSERT_EQ(s.rawLoadCount.load(), static_cast<int64_t>(3));
    s.reset();
    ASSERT_EQ(s.rawLoadPeakNs.load(), static_cast<int64_t>(0));
}

// ---- C3-extent: v4 raw image over the ONE extent file (positioned I/O) ---

namespace {
    // Build an `lb` with `pages` byte-patterned live pages on `m`'s arena.
    void fillArenaPages(gl::LbMemory& lb, int32_t pages, int32_t pb) {
        for (int32_t v = 0; v < pages; ++v) {
            ASSERT_EQ(lb.manager.allocPage(), v);
            std::memset(lb.manager.pageAt(v), vidByte(v),
                        static_cast<std::size_t>(pb));
        }
    }
    // Verify every live vid's first and last byte survived the round trip.
    void checkArenaPages(gl::LbMemory& lb, int32_t pages, int32_t pb) {
        ASSERT_EQ(lb.manager.livePages(), pages);
        for (int32_t v = 0; v < pages; ++v) {
            const unsigned char* page =
                reinterpret_cast<const unsigned char*>(lb.manager.pageAt(v));
            ASSERT_EQ(page[0], vidByte(v));
            ASSERT_EQ(page[pb - 1], vidByte(v));
        }
    }
}

TEST(raw_image, extent_round_trip_positioned_io_at_nonzero_offset) {
    const std::string dir = freshRawDir("extent_rt");
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    gl::LbMemory lb(&m);
    const int32_t pb = m.pageBytes();
    fillArenaPages(lb, 20, pb);

    const std::string chain = "(extent_rt_lb)__";
    const int64_t ordinal = 500001;

    gl::ExtentAllocator alloc;
    alloc.init(m.blockBytes());
    // A filler slab first so the LB's slab lands at a NON-ZERO offset (the
    // offset arithmetic is then a real check, not offset 0).
    const gl::SlabAllocation filler = alloc.allocSlab(m.blockBytes());
    const int64_t need = gl::lbdeload::rawImageBytesFor(lb, chain);
    const gl::SlabAllocation slab = alloc.allocSlab(need);
    ASSERT_TRUE(slab.offset > 0);
    ASSERT_TRUE(need <= slab.classBytes);

    gl::PositionedFile file;
    file.open(std::filesystem::path(dir) / "extent.bin");
    file.preallocate(alloc.highWaterBytes());

    const int64_t payload = gl::lbdeload::dumpLbMemoryRawAt(
        lb, chain, ordinal, file, slab.offset, slab.classBytes);
    ASSERT_EQ(payload, static_cast<int64_t>(20) * pb);

    // Release WITHOUT the container walk (raw-eviction contract), reload.
    lb.manager.releaseAllForRawImage();
    lb.manager.markDeloaded();
    lb.manager.markResident();
    gl::lbdeload::loadLbMemoryRawAt(lb, chain, ordinal, file, slab.offset);
    checkArenaPages(lb, 20, pb);
    (void)filler;
    file.close();
}

TEST(raw_image, extent_in_place_redump_then_class_promotion) {
    // The steady-state (in-place overwrite) and growth (promotion) cases the
    // wiring drives — simulated here directly against the primitives.
    const std::string dir = freshRawDir("extent_grow");
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);                                // 256 KiB block, 8 KiB page
    gl::LbMemory lb(&m);
    const int32_t pb = m.pageBytes();
    const std::string chain = "(extent_grow_lb)__";
    const int64_t ordinal = 500002;

    gl::ExtentAllocator alloc;
    alloc.init(m.blockBytes());
    gl::PositionedFile file;
    file.open(std::filesystem::path(dir) / "extent.bin");

    // 10 pages (~80 KiB + header) -> a 1-block (256 KiB) slab.
    fillArenaPages(lb, 10, pb);
    int64_t need = gl::lbdeload::rawImageBytesFor(lb, chain);
    gl::SlabAllocation slab = alloc.allocSlab(need);
    ASSERT_EQ(slab.classBytes, m.blockBytes());
    file.preallocate(alloc.highWaterBytes());
    gl::lbdeload::dumpLbMemoryRawAt(lb, chain, ordinal, file, slab.offset,
                                    slab.classBytes);
    const int64_t offsetV1 = slab.offset;
    const int64_t fileV1 = file.fileSize();

    // Re-dump the SAME size: fits the class -> IN PLACE (offset unchanged, no
    // allocator interaction, no file growth). This is the near-memcpy hot path.
    need = gl::lbdeload::rawImageBytesFor(lb, chain);
    ASSERT_TRUE(need <= slab.classBytes);
    gl::lbdeload::dumpLbMemoryRawAt(lb, chain, ordinal, file, slab.offset,
                                    slab.classBytes);
    ASSERT_EQ(slab.offset, offsetV1);
    ASSERT_EQ(file.fileSize(), fileV1);

    // Grow the LB past one block: 40 pages (~320 KiB) -> a 2-block class.
    for (int32_t v = 10; v < 40; ++v) {
        ASSERT_EQ(lb.manager.allocPage(), v);
        std::memset(lb.manager.pageAt(v), vidByte(v),
                    static_cast<std::size_t>(pb));
    }
    need = gl::lbdeload::rawImageBytesFor(lb, chain);
    ASSERT_TRUE(need > slab.classBytes);              // outgrew its slab
    // Promote: free the old slab, allocate the bigger class, grow the file.
    alloc.freeSlab(slab.offset, slab.classBytes);
    slab = alloc.allocSlab(need);
    ASSERT_EQ(slab.classBytes, 2 * m.blockBytes());
    ASSERT_NE(slab.offset, offsetV1);                // relocated
    if (alloc.highWaterBytes() > file.fileSize())
        file.grow(alloc.highWaterBytes());
    gl::lbdeload::dumpLbMemoryRawAt(lb, chain, ordinal, file, slab.offset,
                                    slab.classBytes);

    // Reload from the promoted slab: all 40 pages intact.
    lb.manager.releaseAllForRawImage();
    lb.manager.markDeloaded();
    lb.manager.markResident();
    gl::lbdeload::loadLbMemoryRawAt(lb, chain, ordinal, file, slab.offset);
    checkArenaPages(lb, 40, pb);
    file.close();
}

TEST(raw_image, extent_purge_reset_recycles_offsets) {
    const std::string dir = freshRawDir("extent_purge");
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    gl::LbMemory lb(&m);
    const int32_t pb = m.pageBytes();
    const std::string chain = "(extent_purge_lb)__";
    const int64_t ordinal = 500003;

    gl::ExtentAllocator alloc;
    alloc.init(m.blockBytes());
    // Burn some offsets, then purge (reset) -> offsets recycle from 0.
    alloc.allocSlab(3 * m.blockBytes());
    alloc.allocSlab(m.blockBytes());
    ASSERT_TRUE(alloc.highWaterBytes() > 0);
    alloc.reset();
    ASSERT_EQ(alloc.highWaterBytes(), static_cast<int64_t>(0));

    fillArenaPages(lb, 8, pb);
    const int64_t need = gl::lbdeload::rawImageBytesFor(lb, chain);
    const gl::SlabAllocation slab = alloc.allocSlab(need);
    ASSERT_EQ(slab.offset, static_cast<int64_t>(0));   // recycled from 0

    gl::PositionedFile file;
    file.open(std::filesystem::path(dir) / "extent.bin");
    file.preallocate(alloc.highWaterBytes());
    gl::lbdeload::dumpLbMemoryRawAt(lb, chain, ordinal, file, slab.offset,
                                    slab.classBytes);
    lb.manager.releaseAllForRawImage();
    lb.manager.markDeloaded();
    lb.manager.markResident();
    gl::lbdeload::loadLbMemoryRawAt(lb, chain, ordinal, file, slab.offset);
    checkArenaPages(lb, 8, pb);
    file.close();
}

TEST(raw_image, extent_export_release_cycle_dispatches_by_kind) {
    // THE 4 GiB regression (run_accel_fileops_4gib.log): the chapter export's
    // per-theorem G-53 release loop called the v3 releaseStaticBlocks() on an
    // export-reloaded LB whose last dump was an EXTENT raw image — where
    // deloadFiles is empty by design — firing the v3 "without an on-disk
    // image" assert. The fix: the export releases through
    // releaseStaticBlocksDispatch(), the release-side mirror of
    // reloadFromImage's deloadKind dispatch. This test drives the exact
    // export composition: extent evict -> ensureLoadedForRead -> dispatch
    // release -> a later theorem revisits (reload again) -> content intact.
    const std::string dir = freshRawDir("extent_export_cycle");
    gl::staticMemory().openExtentFile(
        std::filesystem::path(dir) / "extent.bin", 64LL << 20);
    {
        gl::Memory m;
        m.setExprKey("(extent_export_cycle_lb)");
        ((m).isActive) = true;
        for (int16_t i = 0; i < 25; ++i)
            ((m).intEncodedStatements).push_back(rawExpr(i));
        for (int32_t i = 0; i < 30; ++i)
            ((m).intLocalEncodedStatementsSet).mint(13 * i + 1);

        m.deloadStaticContainersRaw(dir);           // extent eviction
        ASSERT_TRUE(((m).deloadFiles).empty());          // no named file by design

        m.ensureLoadedForRead(dir);                  // export reload
        ASSERT_TRUE(((m).lbMemory).manager.resident());

        m.releaseStaticBlocksDispatch();             // the FIXED G-53 release
        ASSERT_FALSE(((m).lbMemory).manager.resident());
        ASSERT_EQ(((m).deloadKind), gl::Memory::DeloadKind::Raw);

        // A later theorem revisits the LB: the raw rebind still works — the
        // dispatch skipped the v3 container-release walk, so the bookkeeping
        // survived, and the slab image is still on disk.
        m.ensureLoadedForRead(dir);
        ASSERT_EQ(((m).intEncodedStatements).size(), static_cast<int32_t>(25));
        for (int16_t i = 0; i < 25; ++i)
            ASSERT_EQ(((m).intEncodedStatements)[i].nameId, i);
        for (int32_t i = 0; i < 30; ++i)
            ASSERT_TRUE(((m).intLocalEncodedStatementsSet).contains(13 * i + 1));
    }
    gl::staticMemory().closeExtentFile();
}

TEST(raw_image, named_raw_export_release_cycle_dispatches_by_kind) {
    // The same export composition on the NAMED-file raw path (extent gate
    // off). The dispatch matters here too — beyond the extent assert, a v3
    // release on a Raw-kind LB resets the container bookkeeping
    // (rootVid_/size_/numPages_) that the raw rebind REQUIRES, so the old
    // hardwired v3 release was latently wrong for any raw LB the export
    // reloaded and a later theorem revisited.
    const std::string dir = freshRawDir("named_export_cycle");
    gl::Memory m;
    m.setExprKey("(named_export_cycle_lb)");
    ((m).isActive) = true;
    for (int16_t i = 0; i < 25; ++i)
        ((m).intEncodedStatements).push_back(rawExpr(i));

    m.deloadStaticContainersRaw(dir);               // named raw eviction
    ASSERT_EQ(static_cast<int>(((m).deloadFiles).size()), 1);

    m.ensureLoadedForRead(dir);                      // export reload
    m.releaseStaticBlocksDispatch();                 // routes -> raw release
    ASSERT_FALSE(((m).lbMemory).manager.resident());

    m.ensureLoadedForRead(dir);                      // revisit: rebind works
    ASSERT_EQ(((m).intEncodedStatements).size(), static_cast<int32_t>(25));
    for (int16_t i = 0; i < 25; ++i)
        ASSERT_EQ(((m).intEncodedStatements)[i].nameId, i);
}

TEST(raw_image, canonical_export_release_cycle_dispatches_to_v3) {
    // The dispatch's other leg: a Canonical-kind LB (discharged, or v3-dumped)
    // releases through the v3 walk and reloads through the element-by-element
    // rebuild — the pre-fix export behaviour, preserved exactly.
    gl::deloadStats().reset();
    const std::string dir = freshRawDir("canon_export_cycle");
    gl::Memory m;
    m.setExprKey("(canon_export_cycle_lb)");
    for (int16_t i = 0; i < 20; ++i)
        ((m).intEncodedStatements).push_back(rawExpr(i));

    m.deloadStaticContainers(dir);                   // v3 canonical
    ASSERT_EQ(((m).deloadKind), gl::Memory::DeloadKind::Canonical);
    m.ensureLoadedForRead(dir);                      // export reload (v3)
    m.releaseStaticBlocksDispatch();                 // routes -> v3 release
    ASSERT_FALSE(((m).lbMemory).manager.resident());
    const int64_t rebuilds = gl::deloadStats().indexRebuildCount.load();
    m.ensureLoadedForRead(dir);                      // v3 rebuild path
    ASSERT_TRUE(gl::deloadStats().indexRebuildCount.load() >= rebuilds);
    ASSERT_EQ(((m).intEncodedStatements).size(), static_cast<int32_t>(20));
}

TEST(raw_image, or_matching_canonical_release_discards_derived_views_only) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    const std::string dir = freshRawDir("or_matching_canonical_release");
    gl::Memory memory;
    memory.setExprKey("(or_matching_canonical_release)");
    const auto main = gl::NameMap::MAIN_ID;
    const auto original = ((memory).nameMap).encode("(p[a])");
    const auto complement = ((memory).nameMap).encode("!(p[a])");
    const int64_t key = gl::packInt32Pair(original, main);
    const gl::OrImplicationLeaf leaves[] = {{original, complement}, {complement, original}};
    const int levels[] = {0};
    ((memory).orImplications).insert(key, original, leaves, 2, levels, 1, true);
    gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(analyzer.orImplicationProcessor, memory, ((memory).orImplicationMatches),
        {key, main, 0}, complement);
    ((memory).orImplicationMatches).built = true;
    memory.deloadStaticContainers(dir);
    ASSERT_FALSE(((memory).lbMemory).manager.resident());
    memory.ensureLoadedForRead(dir);
    ASSERT_TRUE(((memory).orImplications).find(key) != nullptr);
    ASSERT_EQ(((memory).orImplicationMatches).liveBytes(), int64_t{0});
    ASSERT_FALSE(((memory).orImplicationMatches).built);
    const int32_t view = gl::tests::OrImplicationProcessorFixture::bindOrCanonicalView(analyzer.orImplicationProcessor, memory,
        ((memory).orImplicationMatches), {key, main, 0}, complement);
    ASSERT_EQ(((memory).orImplicationMatches).liveViews, 1);
    const auto* cache = ((memory).orImplicationMatches).caches().find(
        ((memory).orImplicationMatches).views[view].cacheKey);
    ASSERT_TRUE(cache != nullptr);
    ASSERT_EQ(cache->canonicalId, complement);
}

TEST(raw_image, extent_slab_registry_tracks_alloc_free_and_file_growth) {
    // The manager-level slab bookkeeping behind registry.txt's extent columns:
    // allocateExtentSlab records ordinal->slab, freeExtentSlab drops the row
    // (promotion = free + realloc under the same ordinal), and a bump past the
    // preallocation grows the file in kExtentGrowChunkBytes chunks. Private
    // manager instance (D-139), own extent file, closed at the end.
    const std::string dir = freshRawDir("extent_slab_registry");
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    m.openExtentFile(std::filesystem::path(dir) / "extent.bin", 1 << 20);
    ASSERT_TRUE(m.useExtent());
    ASSERT_TRUE(m.extentSlabRegistry().empty());
    ASSERT_EQ(m.extentFileBytes(), static_cast<int64_t>(1 << 20));

    const gl::SlabAllocation s7 = m.allocateExtentSlab(7, m.blockBytes());
    ASSERT_EQ(m.extentSlabRegistry().size(), static_cast<std::size_t>(1));
    ASSERT_EQ(m.extentSlabRegistry().at(7).offset, s7.offset);
    ASSERT_EQ(m.extentSlabRegistry().at(7).classBytes, s7.classBytes);

    // Class promotion: free + realloc bigger under the SAME ordinal — the
    // registry row is replaced, never duplicated.
    m.freeExtentSlab(7, s7.offset, s7.classBytes);
    ASSERT_TRUE(m.extentSlabRegistry().empty());
    const gl::SlabAllocation s7b =
        m.allocateExtentSlab(7, 3 * m.blockBytes());
    ASSERT_EQ(m.extentSlabRegistry().size(), static_cast<std::size_t>(1));
    ASSERT_EQ(m.extentSlabRegistry().at(7).classBytes, s7b.classBytes);
    // The 4-block class bumped the high-water past the 1 MiB preallocation:
    // the file grew to a whole grow-chunk multiple.
    ASSERT_EQ(m.extentFileBytes(), gl::kExtentGrowChunkBytes);

    // A second ordinal coexists; openExtentFile resets everything.
    m.allocateExtentSlab(9, m.blockBytes());
    ASSERT_EQ(m.extentSlabRegistry().size(), static_cast<std::size_t>(2));
    m.openExtentFile(std::filesystem::path(dir) / "extent.bin", 1 << 20);
    ASSERT_TRUE(m.extentSlabRegistry().empty());
    m.closeExtentFile();
    ASSERT_FALSE(m.useExtent());
}

TEST(raw_image, extent_integration_dump_reload_discharge_via_global_gate) {
    // The pressure-dependent pipeline does not guarantee every INTEGRATION
    // seam is reached: with the global extent path armed, deloadStaticContainersRaw
    // must take the EXTENT branch (no named file, a slab offset), reloadFromImage
    // must dispatch to loadLbMemoryRawAt, a same-size re-dump must overwrite IN
    // PLACE, and discharge must FREE the slab. Opens the global extent file for
    // the test and CLOSES it at the end so every other test still sees the
    // named-file path (test order is unspecified).
    const std::string dir = freshRawDir("extent_integration");
    gl::staticMemory().openExtentFile(
        std::filesystem::path(dir) / "extent.bin", 64LL << 20);
    ASSERT_TRUE(gl::staticMemory().useExtent());
    {
        gl::Memory m;
        m.setExprKey("(extent_integration_lb)");
        ((m).isActive) = true;
        for (int16_t i = 0; i < 40; ++i)
            ((m).intEncodedStatements).push_back(rawExpr(i));
        for (int32_t i = 0; i < 50; ++i)
            ((m).intLocalEncodedStatementsSet).mint(7 * i + 3);

        m.deloadStaticContainersRaw(dir);          // EXTENT branch
        ASSERT_FALSE(((m).lbMemory).manager.resident());
        ASSERT_EQ(((m).deloadKind), gl::Memory::DeloadKind::Raw);
        ASSERT_TRUE(((m).deloadFiles).empty());        // extent path: no named file
        ASSERT_TRUE(((m).rawExtentOffset_) >= 0);
        ASSERT_TRUE(((m).rawExtentClassBytes_) > 0);

        m.ensureLoaded(dir);                        // EXTENT dispatch reload
        ASSERT_TRUE(((m).lbMemory).manager.resident());
        ASSERT_EQ(((m).intEncodedStatements).size(), static_cast<int32_t>(40));
        for (int16_t i = 0; i < 40; ++i)
            ASSERT_EQ(((m).intEncodedStatements)[i].nameId, i);
        for (int32_t i = 0; i < 50; ++i)
            ASSERT_TRUE(((m).intLocalEncodedStatementsSet).contains(7 * i + 3));

        // Same-size re-dump: IN PLACE (offset unchanged, no reallocation).
        const int64_t off = ((m).rawExtentOffset_);
        m.deloadStaticContainersRaw(dir);
        ASSERT_EQ(((m).rawExtentOffset_), off);
        m.ensureLoaded(dir);
        ASSERT_EQ(((m).intEncodedStatements).size(), static_cast<int32_t>(40));

        // Discharge frees the slab (Raw->Canonical, never raw-reloads again).
        gl::GlobalMemoryManager scrPool;
        scrPool.init(kRawCfg16);
        gl::LbArena scratch(&scrPool);
        m.dischargeStatementContent(scratch);
        ASSERT_EQ(((m).rawExtentOffset_), static_cast<int64_t>(-1));
        ASSERT_EQ(((m).rawExtentClassBytes_), 0);
    }
    gl::staticMemory().closeExtentFile();
    ASSERT_FALSE(gl::staticMemory().useExtent());
}

TEST(raw_image, extent_ordinal_tripwire_tracks_slab_occupant) {
    // The stale-occupant tripwire: the header ORDINAL tracks whichever LB last
    // wrote the slab, so a reload with the wrong expected ordinal would fire
    // loadLbMemoryRawAt's `ordinal == expectedOrdinal` assert. That assert
    // ABORTS (the harness cannot catch abort, exactly as the teardown test
    // notes), so the negative is guaranteed structurally; here the POSITIVE is
    // verified — a slab reused by a second LB loads correctly ONLY with the
    // second LB's ordinal, proving the ordinal is genuinely read and matched.
    const std::string dir = freshRawDir("extent_ordinal");
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    const int32_t pb = m.pageBytes();

    gl::ExtentAllocator alloc;
    alloc.init(m.blockBytes());
    gl::PositionedFile file;
    file.open(std::filesystem::path(dir) / "extent.bin");

    // LB-A writes the slab under ordinal 700001.
    gl::LbMemory a(&m);
    fillArenaPages(a, 6, pb);
    const std::string chainA = "(extent_ord_a)__";
    const int64_t need = gl::lbdeload::rawImageBytesFor(a, chainA);
    const gl::SlabAllocation slab = alloc.allocSlab(need);
    file.preallocate(alloc.highWaterBytes());
    gl::lbdeload::dumpLbMemoryRawAt(a, chainA, 700001, file, slab.offset,
                                    slab.classBytes);
    a.manager.releaseAllForRawImage();
    a.manager.markDeloaded();
    a.manager.markResident();
    gl::lbdeload::loadLbMemoryRawAt(a, chainA, 700001, file, slab.offset);
    checkArenaPages(a, 6, pb);

    // LB-B REUSES the same slab (same offset+class) under ordinal 700002; the
    // header now carries 700002, and a load with 700002 succeeds.
    gl::LbMemory b(&m);
    fillArenaPages(b, 6, pb);
    const std::string chainB = "(extent_ord_b)__";
    gl::lbdeload::dumpLbMemoryRawAt(b, chainB, 700002, file, slab.offset,
                                    slab.classBytes);
    b.manager.releaseAllForRawImage();
    b.manager.markDeloaded();
    b.manager.markResident();
    gl::lbdeload::loadLbMemoryRawAt(b, chainB, 700002, file, slab.offset);
    checkArenaPages(b, 6, pb);
    file.close();
}

TEST(raw_image, parked_source_and_pending_receipt_survive_eviction) {
    const std::string dir = freshRawDir("parked_source_state");
    gl::GlobalMemoryManager manager;
    manager.init(gl::StaticMemoryConfig{4 << 20, 1 << 18});
    gl::LbMemory memory(&manager);
    gl::RejectedMapValue record;
    record.sourceExpression = 41;
    record.iteration = 3;
    record.flags = 9;
    record.rewrites = {20,41,30};
    record.levels = {0,5};
    const int64_t key = gl::packInt32Pair(7,3);
    const int64_t receipt = gl::packInt32Pair(41,3);
    ((((memory).overallHashMemory)).rejectedMap).assignRun(key,{record});
    ((memory).existenceReplays).upsert(receipt,25);
    const int64_t bytes = memory.liveBytes();
    const std::string chain = "parked-existence-raw-storage";
    const int64_t ordinal = 900102;
    gl::lbdeload::dumpLbMemoryRaw(memory,chain,ordinal,dir);
    memory.manager.releaseAllForRawImage();
    memory.manager.markDeloaded();
    memory.manager.markResident();
    gl::lbdeload::loadLbMemoryRaw(memory,chain,ordinal,gl::lbdeload::rawFileName(ordinal),dir);
    ASSERT_EQ(*((memory).existenceReplays).find(receipt),25);
    const auto after = ((((memory).overallHashMemory)).rejectedMap).recordAt(
        ((((memory).overallHashMemory)).rejectedMap).lookup(key),0);
    ASSERT_TRUE(gl::Codec<gl::RejectedMapValue>::serialize(after) == gl::Codec<gl::RejectedMapValue>::serialize(record));
    ASSERT_EQ(memory.liveBytes(),bytes);
}


TEST(raw_image, shared_rule_values_keep_private_large_payload_through_both_formats) {
    const std::string directory = freshRawDir("shared_rule_values_large");
    gl::GlobalMemoryManager manager;
    manager.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbMemory source(&manager), restored(&manager);
    gl::ScratchArena scratch(&manager);
    gl::NormKey key{3, {1, 0, 2, 0, 3, 0}};
    gl::OwnerSet large, small;
    large.hasLooseOwner = true;
    for (int32_t owner = 1; owner <= 800; ++owner)
        large.owners.push_back({static_cast<gl::RuleOwner>(owner), -1});
    const auto expected = gl::Codec<gl::OwnerSet>::serialize(large);
    ASSERT_TRUE(expected.size() > 2048);
    ((((source).overallHashMemory)).normalizedEncodedSubkeys).assignRun(key, {large});
    ((((source).workingMemory)).normalizedEncodedSubkeys).assignRun(key, {small});
    source.sharedNativeKeys.collect(scratch);
    const std::string chain = "shared-rule-values";
    const int64_t ordinal = 900111;
    gl::lbdeload::dumpLbMemoryRaw(source, chain, ordinal, directory);
    source.manager.releaseAllForRawImage(); source.manager.markDeloaded(); source.manager.markResident();
    gl::lbdeload::loadLbMemoryRaw(source, chain, ordinal,
        gl::lbdeload::rawFileName(ordinal), directory);
    auto& map = ((((source).overallHashMemory)).normalizedEncodedSubkeys);
    ASSERT_TRUE(gl::Codec<gl::OwnerSet>::serialize(map.recordAt(map.lookup(key), 0)) == expected);
    ((source).workingMemory).resetToFresh();
    source.sharedNativeKeys.collect(scratch);
    ASSERT_TRUE(gl::Codec<gl::OwnerSet>::serialize(map.recordAt(map.lookup(key), 0)) == expected);
    const auto files = gl::lbdeload::dumpLbMemory(source, chain, ordinal + 1, directory, 1 << 18);
    gl::lbdeload::loadLbMemory(restored, chain, files, directory);
    auto& result = ((((restored).overallHashMemory)).normalizedEncodedSubkeys);
    ASSERT_TRUE(gl::Codec<gl::OwnerSet>::serialize(result.recordAt(result.lookup(key), 0)) == expected);
    ASSERT_TRUE(((((restored).workingMemory)).normalizedEncodedSubkeys).empty());
}


TEST(raw_image, shared_expression_handles_and_lifetime_survive_raw_reload) {
    gl::GlobalMemoryManager manager; manager.init(kRawCfg16);
    gl::LbMemory memory(&manager);
    for (int i = 0; i < 300; ++i) {
        ((memory).intEncodedStatements).push_back(rawExpr(i));
        ((memory).intExternalStatements).push_back(rawExpr(i));
    }
    const auto shape = memory.manager.rawShape();
    std::vector<uint8_t> bitmap(static_cast<size_t>((shape.vidCount + 7) / 8));
    memory.manager.fillLiveBitmap(bitmap.data(), static_cast<int64_t>(bitmap.size()));
    const auto bytes = emitToBuffer(memory.manager);
    memory.manager.releaseAllForRawImage(); memory.manager.markDeloaded();
    memory.manager.markResident();
    memory.manager.restoreForRawLoad(shape.vidCount, bitmap.data(), shape.livePages, shape.byteBumpCursor);
    fillFromBuffer(memory.manager, bytes);
    ASSERT_EQ(memory.sharedExpressions.count(), 300);
    ASSERT_TRUE(&((memory).intEncodedStatements)[83] == &((memory).intExternalStatements)[83]);
    ((memory).intEncodedStatements).clear(); ASSERT_EQ(memory.sharedExpressions.count(), 300);
    ((memory).intExternalStatements).erase(83); ASSERT_EQ(memory.sharedExpressions.count(), 299);
    ((memory).intExternalStatements).push_back(rawExpr(83)); ASSERT_EQ(memory.sharedExpressions.count(), 300);
    ((memory).intExternalStatements).clear(); ASSERT_EQ(memory.sharedExpressions.liveBytes(), 0);
}

// ---- Sub-page tier (D-464) ---------------------------

TEST(raw_image, sub_page_lists_survive_a_raw_round_trip) {
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);                         // 8 KiB pages: 32 slots of 256 bytes
    gl::LbArena a(&m);
    a.enableSubPages();
    std::vector<gl::LbArena::SubPage> slots;
    for (int32_t k = 0; k < 40; ++k) {
        slots.push_back(a.allocSubPage(2));
        std::memset(a.pageAt(slots.back().vid) + slots.back().offset, 1 + k, 256);
    }
    const int32_t page = a.allocPage();
    std::memset(a.pageAt(page), 0x5A, static_cast<std::size_t>(a.pageBytes()));
    a.freeSubPage(slots[7], 2);
    a.freeSubPage(slots[30], 2);
    const auto shape = a.rawShape();
    std::vector<unsigned char> bitmap(static_cast<std::size_t>((shape.vidCount + 7) / 8));
    a.fillLiveBitmap(bitmap.data(), static_cast<int32_t>(bitmap.size()));
    const auto bytes = emitToBuffer(a);
    a.releaseAllForRawImage();
    a.markDeloaded();
    a.markResident();
    a.restoreForRawLoad(shape.vidCount, bitmap.data(), shape.livePages,
                        shape.byteBumpCursor);
    // Bound but unfilled pages: only the content-free audit applies.
    a.assertPageTierInvariants();
    fillFromBuffer(a, bytes);
    a.assertSubPageInvariants();
    a.assertInvariants();
    ASSERT_EQ(a.liveSubPages(2), 38);
    ASSERT_EQ(a.subPageSlabs(2), 2);
    for (int32_t k = 0; k < 40; ++k) {
        if (k == 7 || k == 30) continue;
        ASSERT_EQ(static_cast<int32_t>(a.pageAt(slots[k].vid)[slots[k].offset]), 1 + k);
    }
    ASSERT_EQ(static_cast<unsigned char>(a.pageAt(page)[100]), 0x5A);
    // The restored lists hand back exactly the slots that were free.
    const gl::LbArena::SubPage first = a.allocSubPage(2);
    const gl::LbArena::SubPage second = a.allocSubPage(2);
    ASSERT_EQ(first.vid, slots[30].vid);
    ASSERT_EQ(first.offset, slots[30].offset);
    ASSERT_EQ(second.vid, slots[7].vid);
    ASSERT_EQ(second.offset, slots[7].offset);
    a.freeSubPage(first, 2);
    a.freeSubPage(second, 2);
    a.assertInvariants();
}

TEST(raw_image, lb_memory_arena_offers_sub_pages) {
    gl::GlobalMemoryManager m;
    m.init(kRawCfg16);
    gl::LbMemory memory(&m);
    ASSERT_TRUE(memory.manager.subPagesEnabled());
    ((memory).intEncodedStatements).push_back(rawExpr(1));
    int32_t slabs = 0;
    for (int32_t cls = 0; cls < gl::LbArena::kSubPageMaxClasses; ++cls)
        slabs += memory.manager.subPageSlabs(cls);
    ASSERT_TRUE(slabs >= 1);
}
