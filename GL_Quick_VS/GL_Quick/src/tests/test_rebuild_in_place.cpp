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
/// @brief Tests for the in-place blob-run rebuild
///        (`BlobCsrValueStore::rebuildRunsInPlace`, the staging flush's
///        write door) and the two `PagedVector` primitives it added.
///
/// @details
/// 1. Over random rounds of produced (grown) and carried-over keys, with new
///    keys minted after the rebuild, the in-place rebuild
///    (`HashMap::assignAllRunsGenerated`) leaves the map byte-identical to the
///    scratch twin (`HashMap::assignAllRunsViaScratch`) — including a round
///    that produces nothing, a round producing only new keys, and a round
///    producing key 1 (the whole map moves).
/// 2. `PagedVector::growTo` reserves slots across page boundaries and
///    `writeRunAt` fills existing slots.
/// 3. `BlobCsrValueStore::moveShiftedDescending` moves an overlapping range
///    upward with a delta without reading a slot it already wrote.
///
/// Every corpus comes from a fixed-seed linear congruential sequence.

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../prover.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

namespace {

    uint32_t nextRandom(uint32_t& s) {
        s = s * 1664525u + 1013904223u;
        return s >> 16;
    }

    template <class Inner>
    std::vector<char> imageOf(const Inner& m) {
        std::vector<char> out;
        const auto put32 = [&](int32_t v) {
            const char* p = reinterpret_cast<const char*>(&v);
            out.insert(out.end(), p, p + 4);
        };
        put32(m.count());
        for (int32_t id = 1; id <= m.count(); ++id) {
            const gl::StrSpan k = m.keyAt(id);
            put32(k.len);
            out.insert(out.end(), k.ptr, k.ptr + k.len);
            const int32_t rl = m.runLen(id);
            put32(rl);
            for (int32_t j = 0; j < rl; ++j) {
                std::vector<char> b;
                m.blobAt(id, j, b);
                put32(static_cast<int32_t>(b.size()));
                out.insert(out.end(), b.begin(), b.end());
            }
        }
        return out;
    }

    using BlobMap = gl::HashMap<gl::BytesKeyStore, gl::BlobCsrValueStore>;

    // A deterministic appended blob for (round, id, j): the emitter must
    // produce identical records in the count pass and the write pass.
    void appendedBlob(int round, int32_t id, int32_t j, char* out, int32_t& len) {
        len = 1 + ((round * 7 + id * 3 + j) % 13);
        for (int32_t i = 0; i < len; ++i)
            out[i] = static_cast<char>('A' + ((round + id + j + i) % 26));
    }

}  // namespace

// 1. In-place rebuild == scratch rebuild, byte for byte, round after round.
TEST(rebuild_in_place, matches_scratch_twin_over_random_rounds) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 32 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena scratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    BlobMap a(&lbA, &dA);   // in place
    BlobMap b(&lbB, &dB);   // scratch twin

    // Seed both with the same splice sequence over 40 keys.
    uint32_t seed = 20260909u;
    std::vector<std::string> keys;
    for (int k = 0; k < 40; ++k) keys.push_back("key-" + std::to_string(k * 37 % 101));
    for (int t = 0; t < 600; ++t) {
        const int ki = static_cast<int>(nextRandom(seed) % keys.size());
        const gl::StrSpan k(keys[ki].c_str(), static_cast<int32_t>(keys[ki].size()));
        char blob[20];
        const int32_t len = static_cast<int32_t>(nextRandom(seed) % 20u);   // empty blobs too
        for (int32_t i = 0; i < len; ++i) blob[i] = static_cast<char>('a' + (nextRandom(seed) % 26u));
        for (BlobMap* m : { &a, &b }) {
            const int32_t id = m->lookup(k);
            if (id != 0) m->appendBlobToRun(id, blob, len);
            else m->assignRun(k, blob, &len, 1);
        }
    }
    ASSERT_TRUE(imageOf(a) == imageOf(b));
    ASSERT_EQ(a.count(), 40);

    // Rounds: 0 = nothing produced; 1 = only new keys; 2 = key 1 produced;
    // 3..7 = random subsets, growth 1..3 blobs, 0..2 new keys.
    for (int round = 0; round < 8; ++round) {
        const int32_t stored = a.count();
        std::vector<char> produced(static_cast<std::size_t>(stored), 0);
        int32_t newN = 0;
        if (round == 1) newN = 2;
        else if (round == 2) { produced[0] = 1; newN = 1; }
        else if (round >= 3) {
            for (int32_t id = 1; id <= stored; ++id)
                produced[static_cast<std::size_t>(id - 1)] = (nextRandom(seed) % 3u == 0u) ? 1 : 0;
            newN = static_cast<int32_t>(nextRandom(seed) % 3u);
        }
        const int32_t keyCount = stored + newN;
        const auto emitter = [&](BlobMap& m) {
            return [&, round, stored](int32_t id, auto&& sink) {
                if (id > stored) {
                    char blob[16]; int32_t len = 0;
                    appendedBlob(round, id, 0, blob, len);
                    sink.blob(blob, len);
                    appendedBlob(round, id, 1, blob, len);
                    sink.blob(blob, len);
                    return;
                }
                if (produced[static_cast<std::size_t>(id - 1)] == 0) {
                    sink.unchanged(id);
                    return;
                }
                m.forEachBlobContiguous(id, scratch,
                    [&](const char* p, int32_t l) { sink.blob(p, l); });
                const int32_t growth = 1 + ((round + id) % 3);
                for (int32_t j = 0; j < growth; ++j) {
                    char blob[16]; int32_t len = 0;
                    appendedBlob(round, id, j, blob, len);
                    sink.blob(blob, len);
                }
            };
        };
        a.assignAllRunsGenerated(keyCount, emitter(a), scratch);
        b.assignAllRunsViaScratch(keyCount, emitter(b), scratch);
        for (int32_t i = 0; i < newN; ++i) {
            const std::string nk = "new-" + std::to_string(round) + "-" + std::to_string(i);
            const gl::StrSpan k(nk.c_str(), static_cast<int32_t>(nk.size()));
            ASSERT_EQ(a.mint(k), stored + i + 1);
            ASSERT_EQ(b.mint(k), stored + i + 1);
        }
        ASSERT_EQ(a.count(), keyCount);
        ASSERT_TRUE(imageOf(a) == imageOf(b));
    }
    ASSERT_TRUE(a.count() > 43);
}

// 2. growTo reserves across pages; writeRunAt fills existing slots only.
TEST(rebuild_in_place, paged_vector_grow_and_write_run) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 8 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&lb, &d);
    for (int32_t i = 0; i < 10; ++i) v.push_back(i);
    v.growTo(5000);                       // several 8 KiB pages of int32
    ASSERT_EQ(v.size(), 5000);
    ASSERT_TRUE(d == gl::DirtyState::Restructured);
    for (int32_t i = 0; i < 10; ++i) ASSERT_EQ(v[i], i);
    std::vector<int32_t> src(4990);
    for (int32_t i = 0; i < 4990; ++i) src[static_cast<std::size_t>(i)] = 100000 + i;
    v.writeRunAt(10, src.data(), 4990);
    for (int32_t i = 10; i < 5000; ++i) ASSERT_EQ(v[i], 100000 + (i - 10));
    v.growTo(5000);                       // no-op
    ASSERT_EQ(v.size(), 5000);
    v.writeRunAt(0, src.data(), 0);       // empty run is a no-op
    ASSERT_EQ(v[0], 0);
}

// 3. An overlapping upward shifted move over page boundaries.
TEST(rebuild_in_place, move_shifted_descending_overlapping) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 8 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&lb, &d);
    for (int32_t i = 0; i < 6000; ++i) v.push_back(i);
    // Move [1000, 4000) to [1500, 4500) adding 7: overlap of 2500 slots.
    gl::BlobCsrValueStore::moveShiftedDescending(v, 1000, 1500, 3000, 7);
    for (int32_t i = 0; i < 1000; ++i) ASSERT_EQ(v[i], i);
    for (int32_t i = 1500; i < 4500; ++i) ASSERT_EQ(v[i], (i - 500) + 7);
    for (int32_t i = 4500; i < 6000; ++i) ASSERT_EQ(v[i], i);
    // In-place delta (dest == from).
    gl::BlobCsrValueStore::moveShiftedDescending(v, 0, 0, 1000, -1);
    for (int32_t i = 0; i < 1000; ++i) ASSERT_EQ(v[i], i - 1);
    gl::BlobCsrValueStore::moveShiftedDescending(v, 0, 0, 0, 5);   // no-op
    ASSERT_EQ(v[0], -1);
}
