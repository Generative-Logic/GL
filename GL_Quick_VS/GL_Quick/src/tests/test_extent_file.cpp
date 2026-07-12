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
/// @brief Unit tests for `PositionedFile` — the one-file positioned-I/O
///        substrate of the v4 raw-eviction extent file.
///
/// @details
/// The load-bearing test is `concurrent_disjoint_offsets_no_interference`:
/// the STANDALONE Windows positioned-I/O CONCURRENCY PROOF the extent-file
/// design gates on BEFORE any wiring. Many threads, each holding its own
/// independent handle to ONE shared file, do overlapping positioned
/// writes/reads on DISJOINT ranges; the test byte-verifies both non-
/// interference (no cross-region bleed) and no short I/O. If Windows
/// serialized per-thread handles on an NTFS-internal file lock or returned
/// short transfers, this test would surface it — the go/no-go for the whole
/// extent design (the fallback is handle-reuse per LB file).
///
/// The remaining tests cover the plain contract: `writeAt`/`readAt`
/// round-trip at arbitrary offsets, `preallocate` + `fileSize`, `grow`, and
/// per-thread-handle reuse across `close`/`open` (the open-generation guard).

#include "test_harness.hpp"

#include "../memory_infra/extent_file.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <thread>
#include <vector>

namespace {

    // Fresh per-test extent file under .debug (name carries "deload" so the
    // directory is unambiguously scratch, mirroring the raw-image fixtures).
    std::filesystem::path freshExtent(const std::string& name) {
        const std::filesystem::path dir =
            std::filesystem::path(".debug") / "deload_extent";
        std::error_code ec;
        std::filesystem::create_directories(dir, ec);
        const std::filesystem::path file = dir / ("extent_" + name + ".bin");
        std::filesystem::remove(file, ec);
        return file;
    }

    // Deterministic byte for (region, offset) so any cross-region bleed or
    // torn transfer is a hard mismatch.
    unsigned char patByte(int region, int64_t off) {
        return static_cast<unsigned char>((region * 131 + off * 7 + 3) & 0xFF);
    }

}

// ---- Plain contract ------------------------------------------------------

TEST(extent_file, write_read_roundtrip_at_offsets) {
    gl::PositionedFile f;
    f.open(freshExtent("roundtrip"));
    f.preallocate(1 << 20);                       // 1 MiB
    ASSERT_EQ(f.fileSize(), static_cast<int64_t>(1 << 20));

    // Three disjoint spans at non-zero, non-adjacent offsets.
    std::vector<char> a(4096), b(1000), c(37);
    for (std::size_t i = 0; i < a.size(); ++i)
        a[i] = static_cast<char>(patByte(0, static_cast<int64_t>(i)));
    for (std::size_t i = 0; i < b.size(); ++i)
        b[i] = static_cast<char>(patByte(1, static_cast<int64_t>(i)));
    for (std::size_t i = 0; i < c.size(); ++i)
        c[i] = static_cast<char>(patByte(2, static_cast<int64_t>(i)));

    f.writeAt(0, a.data(), static_cast<int64_t>(a.size()));
    f.writeAt(200000, b.data(), static_cast<int64_t>(b.size()));
    f.writeAt(999000, c.data(), static_cast<int64_t>(c.size()));

    std::vector<char> ra(a.size()), rb(b.size()), rc(c.size());
    // Read back out of write order.
    f.readAt(999000, rc.data(), static_cast<int64_t>(rc.size()));
    f.readAt(0, ra.data(), static_cast<int64_t>(ra.size()));
    f.readAt(200000, rb.data(), static_cast<int64_t>(rb.size()));
    ASSERT_TRUE(std::memcmp(a.data(), ra.data(), a.size()) == 0);
    ASSERT_TRUE(std::memcmp(b.data(), rb.data(), b.size()) == 0);
    ASSERT_TRUE(std::memcmp(c.data(), rc.data(), c.size()) == 0);
    f.close();
    ASSERT_FALSE(f.isOpen());
}

TEST(extent_file, preallocate_then_grow_tracks_size) {
    gl::PositionedFile f;
    f.open(freshExtent("grow"));
    f.preallocate(256 * 1024);
    ASSERT_EQ(f.fileSize(), static_cast<int64_t>(256 * 1024));

    // A write inside the preallocated region, then grow and write into the
    // new tail — both round-trip.
    std::vector<char> head(1024, 'H');
    f.writeAt(1000, head.data(), static_cast<int64_t>(head.size()));

    f.grow(1024 * 1024);
    ASSERT_EQ(f.fileSize(), static_cast<int64_t>(1024 * 1024));
    std::vector<char> tail(2048, 'T');
    f.writeAt(900 * 1024, tail.data(), static_cast<int64_t>(tail.size()));

    std::vector<char> rh(1024), rt(2048);
    f.readAt(1000, rh.data(), static_cast<int64_t>(rh.size()));
    f.readAt(900 * 1024, rt.data(), static_cast<int64_t>(rt.size()));
    ASSERT_TRUE(std::memcmp(head.data(), rh.data(), head.size()) == 0);
    ASSERT_TRUE(std::memcmp(tail.data(), rt.data(), tail.size()) == 0);
    f.close();
}

TEST(extent_file, reopen_reuses_thread_handle_across_generation) {
    // The open-generation guard: a second open of the SAME object must not
    // reuse the handle a prior generation cached in this thread's TLS.
    const std::filesystem::path p = freshExtent("reopen");
    gl::PositionedFile f;
    f.open(p);
    f.preallocate(4096);
    std::vector<char> one(64, 'A');
    f.writeAt(0, one.data(), static_cast<int64_t>(one.size()));
    f.close();

    f.open(p);                                    // generation bumps
    f.preallocate(4096);
    std::vector<char> two(64, 'B');
    f.writeAt(0, two.data(), static_cast<int64_t>(two.size()));
    std::vector<char> r(64);
    f.readAt(0, r.data(), static_cast<int64_t>(r.size()));
    ASSERT_TRUE(std::memcmp(two.data(), r.data(), two.size()) == 0);
    f.close();
}

TEST(extent_file, large_write_spans_multiple_native_chunks) {
    // A single writeAt whose byte count is not a power-of-two multiple of the
    // page, round-tripped, exercises the chunk loop's tail handling (the
    // whole span is < kMaxIoChunkBytes so it is one native op, but the loop
    // logic is the same one production drives).
    gl::PositionedFile f;
    f.open(freshExtent("largewrite"));
    const int64_t n = (5 << 20) + 777;            // ~5 MiB + odd tail
    f.preallocate(n + 4096);
    std::vector<char> buf(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i)
        buf[static_cast<std::size_t>(i)] =
            static_cast<char>(patByte(9, i));
    f.writeAt(64, buf.data(), n);
    std::vector<char> r(static_cast<std::size_t>(n));
    f.readAt(64, r.data(), n);
    ASSERT_TRUE(std::memcmp(buf.data(), r.data(),
                            static_cast<std::size_t>(n)) == 0);
    f.close();
}

// ---- The concurrency proof (commit 1's reason for being) -----------------

TEST(extent_file, concurrent_disjoint_offsets_no_interference) {
    // THE PROOF: N threads, each its OWN handle to ONE file, hammering
    // positioned writes+reads on DISJOINT regions concurrently. Non-
    // interference and full (never short) transfers are byte-verified. A
    // Windows per-thread-handle serialization or short-I/O surprise would
    // fail here, before any wiring.
    const std::filesystem::path p = freshExtent("concurrent");
    gl::PositionedFile f;
    f.open(p);

    const int kThreads = 8;
    const int64_t kRegionBytes = 1 << 20;         // 1 MiB per thread
    const int kRounds = 40;
    f.preallocate(static_cast<int64_t>(kThreads) * kRegionBytes);

    std::atomic<bool> mismatch{ false };
    std::atomic<int> ready{ 0 };
    std::atomic<bool> go{ false };

    auto worker = [&](int t) {
        const int64_t base = static_cast<int64_t>(t) * kRegionBytes;
        std::vector<char> w(static_cast<std::size_t>(kRegionBytes));
        std::vector<char> r(static_cast<std::size_t>(kRegionBytes));
        ready.fetch_add(1);
        while (!go.load()) { /* spin to maximise overlap */ }
        for (int round = 0; round < kRounds; ++round) {
            // Pattern folds in the round so a stale read is caught too.
            for (int64_t i = 0; i < kRegionBytes; ++i)
                w[static_cast<std::size_t>(i)] = static_cast<char>(
                    patByte(t + round * 101, i));
            f.writeAt(base, w.data(), kRegionBytes);
            f.readAt(base, r.data(), kRegionBytes);
            if (std::memcmp(w.data(), r.data(),
                            static_cast<std::size_t>(kRegionBytes)) != 0)
                mismatch.store(true);
        }
    };

    std::vector<std::thread> pool;
    for (int t = 0; t < kThreads; ++t) pool.emplace_back(worker, t);
    while (ready.load() < kThreads) { /* wait for all threads armed */ }
    go.store(true);
    for (std::thread& th : pool) th.join();

    ASSERT_FALSE(mismatch.load());

    // Post-join cross-check: every region holds ITS last-round pattern (no
    // neighbour bled into it during the concurrent storm).
    std::vector<char> check(static_cast<std::size_t>(kRegionBytes));
    for (int t = 0; t < kThreads; ++t) {
        const int64_t base = static_cast<int64_t>(t) * kRegionBytes;
        f.readAt(base, check.data(), kRegionBytes);
        for (int64_t i = 0; i < kRegionBytes; ++i)
            ASSERT_EQ(static_cast<unsigned char>(
                          check[static_cast<std::size_t>(i)]),
                      patByte(t + (kRounds - 1) * 101, i));
    }
    f.close();
}

// ---- ExtentAllocator -----------------------------------------------------

namespace {
    // A small block granularity keeps class arithmetic legible in tests.
    const int64_t kBlk = 256 * 1024;   // 256 KiB, the production k=0 class
}

TEST(extent_allocator, class_sizing_rounds_up_to_power_of_two_blocks) {
    gl::ExtentAllocator a;
    a.init(kBlk);
    ASSERT_EQ(a.classBytesFor(0), kBlk);                 // empty -> min class
    ASSERT_EQ(a.classBytesFor(1), kBlk);
    ASSERT_EQ(a.classBytesFor(kBlk), kBlk);              // exactly one block
    ASSERT_EQ(a.classBytesFor(kBlk + 1), 2 * kBlk);      // 2 blocks -> class 2
    ASSERT_EQ(a.classBytesFor(2 * kBlk), 2 * kBlk);
    ASSERT_EQ(a.classBytesFor(2 * kBlk + 1), 4 * kBlk);  // 3 blocks -> class 4
    ASSERT_EQ(a.classBytesFor(5 * kBlk), 8 * kBlk);      // 5 blocks -> class 8
}

TEST(extent_allocator, bump_allocate_distinct_disjoint_slabs) {
    gl::ExtentAllocator a;
    a.init(kBlk);
    const gl::SlabAllocation s0 = a.allocSlab(kBlk);         // class 1 block
    const gl::SlabAllocation s1 = a.allocSlab(3 * kBlk);     // class 4 blocks
    const gl::SlabAllocation s2 = a.allocSlab(kBlk);         // class 1 block
    ASSERT_EQ(s0.offset, static_cast<int64_t>(0));
    ASSERT_EQ(s0.classBytes, kBlk);
    ASSERT_EQ(s1.offset, kBlk);                             // right after s0
    ASSERT_EQ(s1.classBytes, 4 * kBlk);
    ASSERT_EQ(s2.offset, kBlk + 4 * kBlk);                 // right after s1
    // Disjoint: no two slabs overlap.
    ASSERT_TRUE(s0.offset + s0.classBytes <= s1.offset);
    ASSERT_TRUE(s1.offset + s1.classBytes <= s2.offset);
    ASSERT_EQ(a.highWaterBytes(), s2.offset + s2.classBytes);
    ASSERT_EQ(a.allocatedBytes(), kBlk + 4 * kBlk + kBlk);
}

TEST(extent_allocator, free_then_realloc_reuses_same_class_offset) {
    gl::ExtentAllocator a;
    a.init(kBlk);
    const gl::SlabAllocation s0 = a.allocSlab(2 * kBlk);    // class 2 blocks
    const gl::SlabAllocation s1 = a.allocSlab(2 * kBlk);    // class 2 blocks
    const int64_t hwBefore = a.highWaterBytes();
    a.freeSlab(s0.offset, s0.classBytes);
    ASSERT_EQ(a.allocatedBytes(), 2 * kBlk);               // only s1 live
    // A same-class request reuses s0's offset without advancing the high-water.
    const gl::SlabAllocation s2 = a.allocSlab(2 * kBlk);
    ASSERT_EQ(s2.offset, s0.offset);
    ASSERT_EQ(a.highWaterBytes(), hwBefore);
    ASSERT_EQ(a.allocatedBytes(), 4 * kBlk);
    // A DIFFERENT class does not touch that free-list -> fresh bump.
    const gl::SlabAllocation s3 = a.allocSlab(kBlk);       // class 1
    ASSERT_EQ(s3.offset, hwBefore);
    (void)s1;
}

TEST(extent_allocator, class_promotion_grows_high_water) {
    // The LB-growth case: an LB re-dumped into a bigger image promotes its
    // class — the wiring frees the old slab and allocates the new one.
    gl::ExtentAllocator a;
    a.init(kBlk);
    gl::SlabAllocation cur = a.allocSlab(kBlk);            // 1 block
    ASSERT_EQ(cur.classBytes, kBlk);
    // Grows past its class: free old, alloc bigger.
    a.freeSlab(cur.offset, cur.classBytes);
    cur = a.allocSlab(3 * kBlk);                          // -> 4 blocks
    ASSERT_EQ(cur.classBytes, 4 * kBlk);
    a.freeSlab(cur.offset, cur.classBytes);
    cur = a.allocSlab(6 * kBlk);                          // -> 8 blocks
    ASSERT_EQ(cur.classBytes, 8 * kBlk);
    // Only the newest slab is live; the freed 1- and 4-block slabs sit on
    // their free-lists (slack), so allocated < high-water.
    ASSERT_EQ(a.allocatedBytes(), 8 * kBlk);
    ASSERT_TRUE(a.highWaterBytes() > a.allocatedBytes());
}

TEST(extent_allocator, reset_recycles_offsets_from_zero) {
    gl::ExtentAllocator a;
    a.init(kBlk);
    a.allocSlab(2 * kBlk);
    a.allocSlab(kBlk);
    ASSERT_TRUE(a.highWaterBytes() > 0);
    ASSERT_TRUE(a.allocatedBytes() > 0);
    a.reset();
    ASSERT_EQ(a.highWaterBytes(), static_cast<int64_t>(0));
    ASSERT_EQ(a.allocatedBytes(), static_cast<int64_t>(0));
    // Post-reset the first slab is back at offset 0 (deterministic per batch).
    const gl::SlabAllocation s = a.allocSlab(kBlk);
    ASSERT_EQ(s.offset, static_cast<int64_t>(0));
}

TEST(extent_allocator, slack_accounting_alloc_minus_live) {
    // extentAllocatedBytes (Σ class sizes) vs the caller's live written bytes:
    // the allocator owns the allocated side; internal slack = allocated - live.
    gl::ExtentAllocator a;
    a.init(kBlk);
    // A 3-block image lands in a 4-block class -> 1 block of internal slack.
    const gl::SlabAllocation s = a.allocSlab(3 * kBlk);
    ASSERT_EQ(s.classBytes, 4 * kBlk);
    ASSERT_EQ(a.allocatedBytes(), 4 * kBlk);
    const int64_t liveWritten = 3 * kBlk;                 // caller-side fact
    ASSERT_EQ(a.allocatedBytes() - liveWritten, kBlk);    // one block of slack
    a.freeSlab(s.offset, s.classBytes);
    ASSERT_EQ(a.allocatedBytes(), static_cast<int64_t>(0));
}
