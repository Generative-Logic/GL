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
/// @brief Unit tests for the cold-map family's stored-value forms
///        (`PodKeyStore`, `ColdHashMap`, `ColdMultiMap`).
///
/// @details
/// The byte-key interner (`ColdHashSet<BytesKeyStore>` == `ColdStringTable`) is
/// covered by `test_cold_string_table.cpp` through the alias. This file covers
/// the forms added by the "full family now" choice: the POD-key set, the
/// set-once `ColdHashMap`, and the append-to-tail `ColdMultiMap`. It exercises
/// id stability and dedup, value / run round trips, the canonical
/// deload-byte-identity property and index-rebuild-on-reload, `copyFrom`
/// deep-copy, determinism (same insert sequence → same column bytes, history
/// invisible), and the lazy zero-pool-block construction contract.

#include "test_harness.hpp"

#include "../memory_infra/cold_hash_map.hpp"
#include "../memory_infra/cold_string_table.hpp"

#include <map>
#include <set>
#include <string>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block.
    const gl::StaticMemoryConfig kMapTestCfg{ 1 << 20, 1 << 18 };

    using PodSet = gl::ColdHashSet<gl::PodKeyStore<int32_t>>;
    using PodMap = gl::ColdHashMap<gl::PodKeyStore<int32_t>, int64_t>;
    using PodMulti = gl::ColdMultiMap<gl::PodKeyStore<int32_t>, int32_t>;
    using PodSetMap = gl::ColdSetMap<gl::PodKeyStore<int32_t>, int32_t>;
    using PodBlobMap = gl::ColdBlobMap<gl::PodKeyStore<int32_t>>;

    // Concatenate variable-length "records" (here just strings) into one blob
    // stream + per-blob lengths — the shape `HashMap::assignRun` consumes.
    struct BlobBatch {
        std::vector<char> bytes;
        std::vector<int32_t> lens;
    };
    inline BlobBatch makeBlobs(const std::vector<std::string>& blobs) {
        BlobBatch b;
        for (const std::string& s : blobs) {
            b.lens.push_back(static_cast<int32_t>(s.size()));
            b.bytes.insert(b.bytes.end(), s.begin(), s.end());
        }
        return b;
    }
    inline int32_t assignBlobs(PodBlobMap& m, int32_t key,
                               const std::vector<std::string>& blobs) {
        const BlobBatch b = makeBlobs(blobs);
        return m.assignRun(key, b.bytes.data(), b.lens.data(),
                           static_cast<int32_t>(blobs.size()));
    }
    inline std::string readBlobStr(const PodBlobMap& m, int32_t id, int32_t j) {
        std::vector<char> out;
        m.blobAt(id, j, out);
        return std::string(out.begin(), out.end());
    }

    // A stateful per-call comparator — the `orBookkeeping` shape (the heap
    // `ValueIdSet` orders ids by their decoded string via a `ValueInterner*`).
    // Here `rank` stands in for the decoded sort key of a value.
    struct ByRank {
        const int32_t* rank;   // rank[v] is value v's sort key
        bool operator()(int32_t a, int32_t b) const { return rank[a] < rank[b]; }
    };

    // Layout probe for the empty-base-optimization regression guard: a set's
    // non-base members in declaration order. If EmptyValueStore were not elided
    // (it is empty, so the private-base inheritance elides it), a PodSet would
    // be larger than this.
    struct SetLayoutProbe {
        gl::LbArena* arena;
        gl::PodKeyStore<int32_t> ks;
        gl::PagedHashIndex idx;
        uint32_t insertEpoch;
    };
}

TEST(cold_hash_map, pod_set_round_trip) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSet s(&lb, &dirty);
    ASSERT_TRUE(s.empty());

    // Ids are dense insertion order; find-or-mint is idempotent.
    ASSERT_EQ(s.mint(100), 1);
    ASSERT_EQ(s.mint(200), 2);
    ASSERT_EQ(s.mint(100), 1);            // dedup, no new id
    ASSERT_EQ(s.count(), 2);
    ASSERT_TRUE(dirty != gl::DirtyState::Clean);

    // Non-minting probe: hit + miss; positional decode.
    ASSERT_EQ(s.lookup(200), 2);
    ASSERT_EQ(s.lookup(999), 0);
    ASSERT_EQ(s.decode(1), 100);
    ASSERT_EQ(s.decode(2), 200);

    // Index growth over many keys keeps ids stable.
    std::vector<int32_t> ids;
    for (int32_t i = 0; i < 300; ++i) ids.push_back(s.mint(1000 + i));
    for (int32_t i = 0; i < 300; ++i) {
        ASSERT_EQ(s.lookup(1000 + i), ids[static_cast<size_t>(i)]);
        ASSERT_EQ(s.decode(ids[static_cast<size_t>(i)]), 1000 + i);
    }
}

TEST(cold_hash_map, pod_map_insert_find_value) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMap m(&lb, &dirty);

    for (int32_t i = 0; i < 50; ++i) {
        const int32_t id = m.insert(7 * i + 1, 1000000LL + i);
        ASSERT_EQ(id, i + 1);                 // one value per key, id order
    }
    ASSERT_EQ(m.count(), 50);

    for (int32_t i = 0; i < 50; ++i) {
        const int64_t* v = m.find(7 * i + 1);
        ASSERT_TRUE(v != nullptr);
        ASSERT_EQ(*v, 1000000LL + i);
        ASSERT_EQ(m.valueAt(i + 1), 1000000LL + i);
        ASSERT_EQ(m.keyAt(i + 1), 7 * i + 1);
    }
    // A genuine miss is nullptr — a defined query result, not a fallback.
    ASSERT_TRUE(m.find(999999) == nullptr);
    ASSERT_EQ(m.lookup(999999), 0);
}

TEST(cold_hash_map, bytes_keyed_map) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdHashMap<gl::BytesKeyStore, int32_t> m(&lb, &dirty);

    const std::string a = "(=[a,b])";
    const std::string b = "(in[(s[a]),N])";
    const int32_t idA = m.insert(gl::StrSpan(a), 11);
    const int32_t idB = m.insert(gl::StrSpan(b), 22);
    ASSERT_EQ(idA, 1);
    ASSERT_EQ(idB, 2);

    const int32_t* va = m.find(gl::StrSpan(a));
    ASSERT_TRUE(va != nullptr && *va == 11);
    ASSERT_TRUE(m.find(gl::StrSpan(b)) != nullptr && *m.find(gl::StrSpan(b)) == 22);
    ASSERT_TRUE(m.find(gl::StrSpan(std::string("(absent[x])"))) == nullptr);
    ASSERT_EQ(m.keyAt(idA).toStdString(), a);
}

TEST(cold_hash_map, multimap_runs) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMulti mm(&lb, &dirty);

    // Key-grouped append-to-tail: 10 -> {1,2,3}, 20 -> {4,5}, 30 -> {6}.
    ASSERT_EQ(mm.appendToTail(10, 1), 1);
    ASSERT_EQ(mm.appendToTail(10, 2), 1);
    ASSERT_EQ(mm.appendToTail(10, 3), 1);
    ASSERT_EQ(mm.appendToTail(20, 4), 2);     // new key -> new id
    ASSERT_EQ(mm.appendToTail(20, 5), 2);
    ASSERT_EQ(mm.appendToTail(30, 6), 3);

    ASSERT_EQ(mm.count(), 3);
    ASSERT_EQ(mm.valueCount(), 6);
    ASSERT_EQ(mm.runLen(1), 3);
    ASSERT_EQ(mm.runLen(2), 2);
    ASSERT_EQ(mm.runLen(3), 1);
    ASSERT_EQ(mm.valueAt(1, 0), 1);
    ASSERT_EQ(mm.valueAt(1, 2), 3);
    ASSERT_EQ(mm.valueAt(2, 1), 5);
    ASSERT_EQ(mm.valueAt(3, 0), 6);
    ASSERT_EQ(mm.lookup(20), 2);
    ASSERT_TRUE(mm.contains(30));
    ASSERT_FALSE(mm.contains(40));
    ASSERT_EQ(mm.keyAt(2), 20);
}

TEST(cold_hash_map, pod_set_deload_byte_identity_and_index_rebuild) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSet a(&lb, &dirty);
    for (int32_t i = 0; i < 120; ++i) a.mint(3 * i + 5);

    std::vector<char> kb;
    a.keyStore().appendKeyBytes(kb, 0);

    // Reload onto a fresh arena (no index in the stream — rebuilt from keys).
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodSet b(&lb2, &dirty2);
    b.keyStore().bulkLoadKeyBytes(kb.data(), a.count());
    b.rebuildIndex();

    ASSERT_EQ(b.count(), a.count());
    for (int32_t i = 0; i < 120; ++i) {
        ASSERT_EQ(b.lookup(3 * i + 5), a.lookup(3 * i + 5));   // dedup survives
        ASSERT_EQ(b.decode(i + 1), a.decode(i + 1));
    }
    std::vector<char> kb2;
    b.keyStore().appendKeyBytes(kb2, 0);
    ASSERT_TRUE(kb2 == kb);                  // image is a pure function of content
}

TEST(cold_hash_map, pod_map_deload_byte_identity) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMap a(&lb, &dirty);
    for (int32_t i = 0; i < 80; ++i) a.insert(11 * i + 2, 5LL * i + 1);

    std::vector<char> kb, vb;
    a.keyStore().appendKeyBytes(kb, 0);
    a.appendValueBytes(vb, 0);

    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodMap b(&lb2, &dirty2);
    b.keyStore().bulkLoadKeyBytes(kb.data(), a.count());
    b.bulkLoadValueBytes(vb.data(), a.count());
    b.rebuildIndex();

    ASSERT_EQ(b.count(), a.count());
    for (int32_t i = 0; i < 80; ++i) {
        const int64_t* v = b.find(11 * i + 2);
        ASSERT_TRUE(v != nullptr);
        ASSERT_EQ(*v, 5LL * i + 1);
    }
    std::vector<char> kb2, vb2;
    b.keyStore().appendKeyBytes(kb2, 0);
    b.appendValueBytes(vb2, 0);
    ASSERT_TRUE(kb2 == kb);
    ASSERT_TRUE(vb2 == vb);
}

TEST(cold_hash_map, multimap_deload_byte_identity) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMulti a(&lb, &dirty);
    // 25 keys, key i carrying (i % 4) + 1 values, all key-grouped (tail append).
    for (int32_t k = 0; k < 25; ++k) {
        const int32_t runs = (k % 4) + 1;
        for (int32_t j = 0; j < runs; ++j) a.appendToTail(100 + k, k * 10 + j);
    }

    std::vector<char> kb, rb, vb;
    a.keyStore().appendKeyBytes(kb, 0);
    a.appendRunStartBytes(rb, 0);
    a.appendValueBytes(vb, 0);

    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodMulti b(&lb2, &dirty2);
    b.keyStore().bulkLoadKeyBytes(kb.data(), a.count());
    b.bulkLoadRunStartBytes(rb.data(), a.count());
    b.bulkLoadValueBytes(vb.data(), a.valueCount());
    b.rebuildIndex();

    ASSERT_EQ(b.count(), a.count());
    ASSERT_EQ(b.valueCount(), a.valueCount());
    for (int32_t id = 1; id <= a.count(); ++id) {
        ASSERT_EQ(b.runLen(id), a.runLen(id));
        for (int32_t j = 0; j < a.runLen(id); ++j)
            ASSERT_EQ(b.valueAt(id, j), a.valueAt(id, j));
        ASSERT_EQ(b.lookup(b.keyAt(id)), a.lookup(a.keyAt(id)));
    }
    std::vector<char> kb2, rb2, vb2;
    b.keyStore().appendKeyBytes(kb2, 0);
    b.appendRunStartBytes(rb2, 0);
    b.appendValueBytes(vb2, 0);
    ASSERT_TRUE(kb2 == kb);
    ASSERT_TRUE(rb2 == rb);
    ASSERT_TRUE(vb2 == vb);
}

TEST(cold_hash_map, hash_index_growth_and_reload) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSet s(&lb, &dirty);
    // Mint enough to drive several throw-away-bucket rehashes (the index grows
    // by reset + re-place at 1/2 load).
    const int32_t N = 500;
    for (int32_t i = 0; i < N; ++i) {
        ASSERT_EQ(s.mint(7 * i + 3), i + 1);   // dense insertion-order ids
        ASSERT_EQ(s.mint(7 * i + 3), i + 1);   // immediate dedup
    }
    ASSERT_EQ(s.count(), N);
    for (int32_t i = 0; i < N; ++i) {
        ASSERT_EQ(s.lookup(7 * i + 3), i + 1);
        ASSERT_EQ(s.decode(i + 1), 7 * i + 3);
    }
    ASSERT_EQ(s.lookup(999999), 0);            // absent

    // Reload onto a fresh arena: the throw-away hash index is rebuilt from the
    // keys (no index in the byte stream) — dedup must survive.
    std::vector<char> kb;
    s.keyStore().appendKeyBytes(kb, 0);
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodSet r(&lb2, &dirty2);
    r.keyStore().bulkLoadKeyBytes(kb.data(), s.count());
    r.rebuildIndex();
    ASSERT_EQ(r.count(), N);
    for (int32_t i = 0; i < N; ++i) ASSERT_EQ(r.lookup(7 * i + 3), i + 1);
    // After reload, a duplicate still dedups and a new key extends in order.
    ASSERT_EQ(r.mint(3), 1);                   // 7*0+3, already present
    ASSERT_EQ(r.count(), N);
    ASSERT_EQ(r.mint(123456), N + 1);
}

TEST(paged_hash_index, slots_in_place_write_overwrite_and_pages) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::PagedHashIndex idx(&lb);
    ASSERT_TRUE(idx.empty());

    // Capacity spanning several pages exercises the directory + page crossing.
    const int32_t slotsPerPage = g.pageBytes() / 4;
    int32_t cap = 1;
    while (cap < slotsPerPage * 3) cap <<= 1;   // power of two, >= 3 pages
    idx.reset(cap);
    ASSERT_EQ(idx.capacity(), cap);
    ASSERT_FALSE(idx.empty());
    // A freshly reset array reads all-zero (reused pages must be cleared).
    ASSERT_EQ(idx.at(0), 0);
    ASSERT_EQ(idx.at(slotsPerPage), 0);
    ASSERT_EQ(idx.at(cap - 1), 0);

    // In-place writes across page boundaries, read back.
    idx.set(0, 11);
    idx.set(slotsPerPage, 22);                  // first slot of the next page
    idx.set(cap - 1, 33);                       // last slot
    ASSERT_EQ(idx.at(0), 11);
    ASSERT_EQ(idx.at(slotsPerPage), 22);
    ASSERT_EQ(idx.at(cap - 1), 33);
    // In-place OVERWRITE — the operation PagedVector forbids.
    idx.set(0, 99);
    ASSERT_EQ(idx.at(0), 99);

    // reset re-zeros and re-sizes, even reusing freed pages.
    idx.reset(64);
    ASSERT_EQ(idx.capacity(), 64);
    for (int32_t i = 0; i < 64; ++i) ASSERT_EQ(idx.at(i), 0);
    idx.clear();
    ASSERT_TRUE(idx.empty());
}

TEST(cold_hash_map, copy_from_deep_copy) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMap src(&lb, &dirty);
    for (int32_t i = 0; i < 30; ++i) src.insert(2 * i + 1, 100LL + i);

    gl::LbArena lbClone(&g);
    gl::DirtyState dirtyClone = gl::DirtyState::Clean;
    PodMap clone(&lbClone, &dirtyClone);
    clone.copyFrom(src);
    ASSERT_EQ(clone.count(), src.count());
    for (int32_t i = 0; i < 30; ++i) {
        const int64_t* v = clone.find(2 * i + 1);
        ASSERT_TRUE(v != nullptr && *v == 100LL + i);
    }
    // The clone inserts independently; the source is unchanged.
    clone.insert(99999, 7LL);
    ASSERT_EQ(clone.lookup(99999), clone.count());
    ASSERT_EQ(src.lookup(99999), 0);
}

TEST(cold_hash_map, determinism_history_invisible) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);

    // Canonical content dumped from a clean set.
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodSet clean(&lb, &d);
    for (int32_t i = 0; i < 64; ++i) clean.mint(9 * i + 4);
    std::vector<char> cleanBytes;
    clean.keyStore().appendKeyBytes(cleanBytes, 0);

    // Same canonical content reached after junk + reset → byte-identical image.
    gl::LbArena lb2(&g);
    gl::DirtyState d2 = gl::DirtyState::Clean;
    PodSet churned(&lb2, &d2);
    for (int32_t i = 0; i < 20; ++i) churned.mint(-(i + 1));   // junk
    churned.resetToFresh();
    for (int32_t i = 0; i < 64; ++i) churned.mint(9 * i + 4);
    std::vector<char> churnedBytes;
    churned.keyStore().appendKeyBytes(churnedBytes, 0);

    ASSERT_TRUE(churnedBytes == cleanBytes);
}

TEST(cold_hash_map, lazy_zero_pool_blocks) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;

    // Constructing every form consumes no pool blocks (lazy geometry).
    PodSet s(&lb, &dirty);
    PodMap m(&lb, &dirty);
    PodMulti mm(&lb, &dirty);
    gl::ColdHashMap<gl::BytesKeyStore, int32_t> bm(&lb, &dirty);
    ASSERT_EQ(g.blocksInUse(), 0);
    ASSERT_EQ(lb.blocksHeld(), 0);
}

TEST(cold_hash_map, unified_hashmap_direct_and_alias_identity) {
    // The three family names are aliases of the single HashMap class.
    static_assert(std::is_same<PodSet,
        gl::HashMap<gl::PodKeyStore<int32_t>, gl::EmptyValueStore>>::value,
        "ColdHashSet must alias HashMap<KeyStore, EmptyValueStore>");
    static_assert(std::is_same<PodMap,
        gl::HashMap<gl::PodKeyStore<int32_t>,
                    gl::SingleValueStore<int64_t>>>::value,
        "ColdHashMap must alias HashMap<KeyStore, SingleValueStore<V>>");
    static_assert(std::is_same<PodMulti,
        gl::HashMap<gl::PodKeyStore<int32_t>,
                    gl::CsrValueStore<int32_t>>>::value,
        "ColdMultiMap must alias HashMap<KeyStore, CsrValueStore<V>>");

    // Drive the class directly through its real name.
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::HashMap<gl::PodKeyStore<int32_t>, gl::SingleValueStore<int64_t>> m(
        &lb, &dirty);
    ASSERT_EQ(m.insert(42, 4242LL), 1);
    ASSERT_EQ(m.insert(43, 4343LL), 2);
    const int64_t* v = m.find(42);
    ASSERT_TRUE(v != nullptr && *v == 4242LL);
    ASSERT_TRUE(m.find(7) == nullptr);
    ASSERT_EQ(m.valueAt(2), 4343LL);
    ASSERT_EQ(m.count(), 2);
}

TEST(cold_hash_map, empty_value_store_is_zero_cost) {
    // EmptyValueStore must stay empty so the private-base inheritance elides it
    // (empty-base optimization). [[no_unique_address]] is a no-op under the
    // project's C++17/MSVC, so EBO is the portable zero-cost form; if this fails
    // a set grew and ColdStringTable's layout is no longer byte-identical.
    static_assert(std::is_empty<gl::EmptyValueStore>::value,
        "EmptyValueStore must stay empty for empty-base optimization");
    static_assert(sizeof(PodSet) == sizeof(SetLayoutProbe),
        "EmptyValueStore must add no bytes to a set (empty-base optimization)");
    ASSERT_TRUE(std::is_empty<gl::EmptyValueStore>::value);
}

TEST(cold_hash_map, byte_set_facets_resolve_and_round_trip) {
    // The byte-key set's nested deload facets (LengthsView / BytesView) must
    // resolve through the ColdHashSet alias (this is the ColdStringTable surface
    // lb_memory.hpp names) and round-trip the two GLDL tags.
    using ByteSet = gl::ColdHashSet<gl::BytesKeyStore>;
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    ByteSet a(&lb, &dirty);
    a.intern(gl::StrSpan(std::string("alpha")));
    a.intern(gl::StrSpan(std::string("beta")));
    a.intern(gl::StrSpan(std::string("")));     // empty key is legal
    a.intern(gl::StrSpan(std::string("gamma")));

    ByteSet::LengthsView aLen(&a);
    ByteSet::BytesView aBytes(&a);
    std::vector<char> lens, bytes;
    aLen.appendSpanBytes(lens, 0);
    aBytes.appendSpanBytes(bytes, 0);
    ASSERT_EQ(aLen.size(), 4);

    // Reload into a fresh set through the facets (lengths staged, bytes joined).
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    ByteSet b(&lb2, &dirty2);
    ByteSet::LengthsView bLen(&b);
    ByteSet::BytesView bBytes(&b);
    bLen.bulkAppendBytes(lens.data(), aLen.size());
    bBytes.bulkAppendBytes(bytes.data(), aBytes.size());

    ASSERT_EQ(b.count(), a.count());
    const std::string gamma = "gamma";
    const gl::StrSpan gammaSpan(gamma);
    const int32_t gammaId = a.lookup(gammaSpan);
    ASSERT_EQ(b.lookup(gammaSpan), gammaId);
    ASSERT_EQ(a.keyStore().hashStored(gammaId),
              a.keyStore().hashProbe(gammaSpan));
    ASSERT_EQ(b.keyStore().hashStored(gammaId),
              b.keyStore().hashProbe(gammaSpan));

    // Cross-arena copy carries the same static location-record digest and
    // rebuilds a lookup index that resolves the same id.
    gl::LbArena lb3(&g);
    gl::DirtyState dirty3 = gl::DirtyState::Clean;
    ByteSet clone(&lb3, &dirty3);
    clone.copyFrom(a);
    ASSERT_EQ(clone.lookup(gammaSpan), gammaId);
    ASSERT_EQ(clone.keyStore().hashStored(gammaId),
              clone.keyStore().hashProbe(gammaSpan));
    std::vector<char> lens2, bytes2;
    ByteSet::LengthsView(&b).appendSpanBytes(lens2, 0);
    ByteSet::BytesView(&b).appendSpanBytes(bytes2, 0);
    ASSERT_TRUE(lens2 == lens);
    ASSERT_TRUE(bytes2 == bytes);
}

TEST(cold_hash_map, pod_set_erase_and_erase_if) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSet s(&lb, &dirty);
    for (int32_t i = 0; i < 20; ++i) s.mint(i);            // keys 0..19, ids 1..20

    // Single-key erase: a hit removes, a miss is a defined false.
    ASSERT_TRUE(s.erase(5));
    ASSERT_FALSE(s.contains(5));
    ASSERT_FALSE(s.erase(5));                              // already gone
    ASSERT_FALSE(s.erase(999));                            // never present
    ASSERT_EQ(s.count(), 19);
    // Every survivor is still found — the index was rebuilt after the shift.
    for (int32_t i = 0; i < 20; ++i)
        if (i != 5) ASSERT_TRUE(s.contains(i));

    // Bulk eraseIf: drop the evens.
    const int32_t removed = s.eraseIf([](int32_t k) { return (k % 2) == 0; });
    ASSERT_EQ(removed, 10);                                // 0,2,...,18
    for (int32_t i = 0; i < 20; ++i) {
        const bool live = (i % 2 == 1) && (i != 5);
        ASSERT_EQ(s.contains(i), live);
    }
    ASSERT_EQ(s.count(), 9);                               // odds 1..19 minus 5
    // A fresh mint extends in order and still dedups.
    const int32_t id = s.mint(777);
    ASSERT_EQ(id, s.count());
    ASSERT_EQ(s.mint(777), id);
}

TEST(cold_hash_map, pod_map_erase_keeps_values_aligned) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMap m(&lb, &dirty);
    for (int32_t i = 0; i < 10; ++i) m.insert(2 * i, 1000LL + i);   // 2i -> 1000+i

    // eraseIf in the middle (key 6, i==3): the value column stays aligned.
    ASSERT_EQ(m.eraseIf([](int32_t k) { return k == 6; }), 1);
    ASSERT_TRUE(m.find(6) == nullptr);
    ASSERT_EQ(m.count(), 9);
    for (int32_t i = 0; i < 10; ++i) {
        if (i == 3) continue;
        const int64_t* v = m.find(2 * i);
        ASSERT_TRUE(v != nullptr && *v == 1000LL + i);
    }
    // Single-key erase + miss.
    ASSERT_TRUE(m.erase(0));
    ASSERT_FALSE(m.erase(0));
    ASSERT_EQ(m.count(), 8);

    // Deload byte-identity after erase: dump, reload, re-dump.
    std::vector<char> kb, vb;
    m.keyStore().appendKeyBytes(kb, 0);
    m.appendValueBytes(vb, 0);
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodMap r(&lb2, &dirty2);
    r.keyStore().bulkLoadKeyBytes(kb.data(), m.count());
    r.bulkLoadValueBytes(vb.data(), m.count());
    r.rebuildIndex();
    for (int32_t i = 1; i < 10; ++i) {                    // keys 0 and 6 gone
        if (i == 3) continue;
        const int64_t* v = r.find(2 * i);
        ASSERT_TRUE(v != nullptr && *v == 1000LL + i);
    }
    std::vector<char> kb2, vb2;
    r.keyStore().appendKeyBytes(kb2, 0);
    r.appendValueBytes(vb2, 0);
    ASSERT_TRUE(kb2 == kb);
    ASSERT_TRUE(vb2 == vb);
}

TEST(cold_hash_map, pod_map_set_value_at_in_place_update) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMap m(&lb, &dirty);
    for (int32_t i = 0; i < 16; ++i) m.insert(i, 100LL + i);
    const int32_t id = m.lookup(7);
    ASSERT_TRUE(id != 0);
    ASSERT_EQ(*m.find(7), 107LL);

    dirty = gl::DirtyState::Clean;                        // observe the escalation
    m.setValueAt(id, 999LL);                              // in-place overwrite
    ASSERT_EQ(static_cast<int>(dirty),
              static_cast<int>(gl::DirtyState::Restructured));
    ASSERT_EQ(*m.find(7), 999LL);
    ASSERT_EQ(m.valueAt(id), 999LL);
    ASSERT_EQ(*m.find(8), 108LL);                         // other keys untouched

    // Deload reflects the updated value.
    std::vector<char> kb, vb;
    m.keyStore().appendKeyBytes(kb, 0);
    m.appendValueBytes(vb, 0);
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodMap r(&lb2, &dirty2);
    r.keyStore().bulkLoadKeyBytes(kb.data(), m.count());
    r.bulkLoadValueBytes(vb.data(), m.count());
    r.rebuildIndex();
    ASSERT_EQ(*r.find(7), 999LL);
}

TEST(cold_hash_map, erase_if_determinism_history_invisible) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);

    // Reach {odds} by inserting 0..30 then erasing the evens...
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodSet a(&lb, &d);
    for (int32_t i = 0; i <= 30; ++i) a.mint(i);
    a.eraseIf([](int32_t k) { return (k % 2) == 0; });
    std::vector<char> ab;
    a.keyStore().appendKeyBytes(ab, 0);

    // ...vs inserting only the odds. eraseIf preserves survivor order, so the
    // canonical key column is byte-identical (erase history invisible).
    gl::LbArena lb2(&g);
    gl::DirtyState d2 = gl::DirtyState::Clean;
    PodSet b(&lb2, &d2);
    for (int32_t i = 1; i <= 30; i += 2) b.mint(i);
    std::vector<char> bb;
    b.keyStore().appendKeyBytes(bb, 0);

    ASSERT_EQ(a.count(), b.count());
    ASSERT_TRUE(ab == bb);
}

TEST(cold_hash_map, pod_set_and_map_facets_round_trip) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;

    // POD set: one KEY tag through KeysView (rebuilds the index on reload).
    PodSet s(&lb, &dirty);
    for (int32_t i = 0; i < 40; ++i) s.mint(5 * i + 1);
    PodSet::KeysView sKeys(&s);
    std::vector<char> sk;
    sKeys.appendSpanBytes(sk, 0);
    ASSERT_EQ(sKeys.size(), 40);

    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodSet s2(&lb2, &dirty2);
    PodSet::KeysView(&s2).bulkAppendBytes(sk.data(), sKeys.size());
    ASSERT_EQ(s2.count(), 40);
    for (int32_t i = 0; i < 40; ++i) ASSERT_EQ(s2.lookup(5 * i + 1), i + 1);

    // POD map: KEY tag (KeysView) then VALUE tag (ValuesView) — key tag first.
    PodMap m(&lb, &dirty);
    for (int32_t i = 0; i < 30; ++i) m.insert(3 * i + 2, 7LL * i + 9);
    PodMap::KeysView mKeys(&m);
    PodMap::ValuesView mVals(&m);
    std::vector<char> mk, mv;
    mKeys.appendSpanBytes(mk, 0);
    mVals.appendSpanBytes(mv, 0);
    ASSERT_EQ(mKeys.size(), 30);
    ASSERT_EQ(mVals.size(), 30);

    gl::LbArena lb3(&g);
    gl::DirtyState dirty3 = gl::DirtyState::Clean;
    PodMap m2(&lb3, &dirty3);
    PodMap::KeysView(&m2).bulkAppendBytes(mk.data(), mKeys.size());
    PodMap::ValuesView(&m2).bulkAppendBytes(mv.data(), mVals.size());
    ASSERT_EQ(m2.count(), 30);
    for (int32_t i = 0; i < 30; ++i) {
        const int64_t* v = m2.find(3 * i + 2);
        ASSERT_TRUE(v != nullptr && *v == 7LL * i + 9);
    }
    // The key facet's clear() resets the whole container (it owns the lifecycle).
    PodMap::KeysView(&m2).clear();
    ASSERT_TRUE(m2.empty());
}

TEST(cold_hash_map, erase_if_compaction_multipage_byte_identical) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);

    // A map whose key + value columns span several pages.
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodMap m(&lb, &d);
    const int32_t N = 4000;
    for (int32_t i = 0; i < N; ++i) m.insert(i, 100LL + i);

    // Erase every key divisible by 3 (scattered across pages) in one pass.
    const int32_t removed = m.eraseIf([](int32_t k) { return k % 3 == 0; });
    int32_t expectRemoved = 0;
    for (int32_t i = 0; i < N; ++i) if (i % 3 == 0) ++expectRemoved;
    ASSERT_EQ(removed, expectRemoved);
    ASSERT_EQ(m.count(), N - expectRemoved);

    // Survivors keep their values and resolve through the rebuilt index;
    // erased keys are gone.
    for (int32_t i = 0; i < N; ++i) {
        const int64_t* v = m.find(i);
        if (i % 3 == 0) ASSERT_TRUE(v == nullptr);
        else ASSERT_TRUE(v != nullptr && *v == 100LL + i);
    }

    // The compacted columns are byte-identical to a from-scratch insert of the
    // survivors in order (the canonical-bytes guarantee of compacting erase).
    gl::LbArena lb2(&g);
    gl::DirtyState d2 = gl::DirtyState::Clean;
    PodMap fresh(&lb2, &d2);
    for (int32_t i = 0; i < N; ++i) if (i % 3 != 0) fresh.insert(i, 100LL + i);
    std::vector<char> mk, mv, fk, fv;
    m.keyStore().appendKeyBytes(mk, 0);
    m.appendValueBytes(mv, 0);
    fresh.keyStore().appendKeyBytes(fk, 0);
    fresh.appendValueBytes(fv, 0);
    ASSERT_TRUE(mk == fk);
    ASSERT_TRUE(mv == fv);
}

TEST(cold_hash_map, set_map_alias_identity) {
    // ColdSetMap is the fourth value-store instantiation of the one HashMap.
    static_assert(std::is_same<PodSetMap,
        gl::HashMap<gl::PodKeyStore<int32_t>,
                    gl::SetValueStore<int32_t>>>::value,
        "ColdSetMap must alias HashMap<KeyStore, SetValueStore<V>>");
    ASSERT_TRUE(true);
}

TEST(cold_hash_map, set_map_insert_sorted_interior_and_dedup) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSetMap m(&lb, &dirty);

    // Key 10's set built out of order stays sorted-unique: {5,1,3,3} -> {1,3,5}.
    ASSERT_EQ(m.insertSorted(10, 5), 1);     // new key, id 1
    ASSERT_EQ(m.insertSorted(10, 1), 1);     // interior splice to the front
    ASSERT_EQ(m.insertSorted(10, 3), 1);     // interior splice to the middle
    ASSERT_EQ(m.insertSorted(10, 3), 1);     // duplicate -> no-op
    ASSERT_EQ(m.runLen(1), 3);
    ASSERT_EQ(m.valueAt(1, 0), 1);
    ASSERT_EQ(m.valueAt(1, 1), 3);
    ASSERT_EQ(m.valueAt(1, 2), 5);

    // A second key, then grow the FIRST key's run again — an interior insert
    // (key 10 is no longer the last key), shifting key 20's run start.
    ASSERT_EQ(m.insertSorted(20, 9), 2);     // new key, id 2
    ASSERT_EQ(m.insertSorted(10, 4), 1);     // {1,3,5} -> {1,3,4,5}
    ASSERT_EQ(m.runLen(1), 4);
    ASSERT_EQ(m.valueAt(1, 0), 1);
    ASSERT_EQ(m.valueAt(1, 1), 3);
    ASSERT_EQ(m.valueAt(1, 2), 4);
    ASSERT_EQ(m.valueAt(1, 3), 5);
    ASSERT_EQ(m.runLen(2), 1);               // key 20's run intact
    ASSERT_EQ(m.valueAt(2, 0), 9);
    ASSERT_EQ(m.valueCount(), 5);

    // Membership probe (linear over the run) hits and misses.
    ASSERT_TRUE(m.setContains(10, 4));
    ASSERT_FALSE(m.setContains(10, 2));
    ASSERT_TRUE(m.setContains(20, 9));
    ASSERT_FALSE(m.setContains(30, 9));      // absent key
}

TEST(cold_hash_map, set_map_custom_comparator_stateful) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSetMap m(&lb, &dirty);

    // Order values by an external rank (the orBookkeeping decoded-id shape).
    // rank[v]: 1->30, 2->10, 3->20 -> sorted run is {2,3,1}.
    int32_t rank[4] = { 0, 30, 10, 20 };
    ByRank cmp{ rank };
    m.insertSorted(7, 1, cmp);
    m.insertSorted(7, 2, cmp);
    m.insertSorted(7, 3, cmp);
    m.insertSorted(7, 2, cmp);               // dedup under the comparator
    ASSERT_EQ(m.runLen(1), 3);
    ASSERT_EQ(m.valueAt(1, 0), 2);           // rank 10
    ASSERT_EQ(m.valueAt(1, 1), 3);           // rank 20
    ASSERT_EQ(m.valueAt(1, 2), 1);           // rank 30
}

TEST(cold_hash_map, set_map_assign_set_replace) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSetMap m(&lb, &dirty);

    // Whole-set assign for a new key, then a trailing key to force interior
    // replaces on the first.
    const int32_t a[3] = { 2, 4, 6 };
    ASSERT_EQ(m.assignSet(100, a, 3), 1);
    const int32_t tail[2] = { 7, 8 };
    ASSERT_EQ(m.assignSet(200, tail, 2), 2);

    // Same-size replace (in place, no shift): {2,4,6} -> {1,4,9}.
    const int32_t same[3] = { 1, 4, 9 };
    m.assignSet(100, same, 3);
    ASSERT_EQ(m.runLen(1), 3);
    ASSERT_EQ(m.valueAt(1, 0), 1);
    ASSERT_EQ(m.valueAt(1, 2), 9);
    ASSERT_EQ(m.valueAt(2, 0), 7);           // tail key still aligned
    ASSERT_EQ(m.valueAt(2, 1), 8);

    // Grow replace: {1,4,9} -> {0,1,2,3,4} (interior, shifts the tail key).
    const int32_t big[5] = { 0, 1, 2, 3, 4 };
    m.assignSet(100, big, 5);
    ASSERT_EQ(m.runLen(1), 5);
    for (int32_t j = 0; j < 5; ++j) ASSERT_EQ(m.valueAt(1, j), j);
    ASSERT_EQ(m.runLen(2), 2);
    ASSERT_EQ(m.valueAt(2, 0), 7);
    ASSERT_EQ(m.valueAt(2, 1), 8);

    // Shrink replace: {0,1,2,3,4} -> {5} (interior, shifts the tail key back).
    const int32_t small[1] = { 5 };
    m.assignSet(100, small, 1);
    ASSERT_EQ(m.runLen(1), 1);
    ASSERT_EQ(m.valueAt(1, 0), 5);
    ASSERT_EQ(m.runLen(2), 2);
    ASSERT_EQ(m.valueAt(2, 0), 7);
    ASSERT_EQ(m.valueAt(2, 1), 8);
    ASSERT_EQ(m.valueCount(), 3);
}

TEST(cold_hash_map, set_map_erase_set_and_erase_if) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSetMap m(&lb, &dirty);

    // Keys 0..9, key k carrying {k, k+100}.
    for (int32_t k = 0; k < 10; ++k) {
        m.insertSorted(k, k);
        m.insertSorted(k, k + 100);
    }
    ASSERT_EQ(m.count(), 10);
    ASSERT_EQ(m.valueCount(), 20);

    // A no-match eraseSetIf is a pure no-op (no dirty escalation).
    dirty = gl::DirtyState::Clean;
    ASSERT_EQ(m.eraseSetIf([](int32_t k) { return k > 1000; }), 0);
    ASSERT_EQ(static_cast<int>(dirty), static_cast<int>(gl::DirtyState::Clean));

    // Single-key erase drops the key and its whole run.
    ASSERT_TRUE(m.eraseSet(4));
    ASSERT_FALSE(m.eraseSet(4));              // already gone -> defined false
    ASSERT_EQ(m.count(), 9);
    ASSERT_EQ(m.valueCount(), 18);
    ASSERT_FALSE(m.setContains(4, 4));

    // Bulk eraseIf: drop the evens; survivors keep their runs aligned.
    const int32_t removed = m.eraseSetIf([](int32_t k) { return (k % 2) == 0; });
    ASSERT_EQ(removed, 4);                    // 0,2,6,8 (4 already gone)
    ASSERT_EQ(m.count(), 5);                  // odds 1,3,5,7,9
    for (int32_t k = 0; k < 10; ++k) {
        const bool live = (k % 2 == 1);
        const int32_t id = m.lookup(k);
        if (!live) { ASSERT_EQ(id, 0); continue; }
        ASSERT_TRUE(id != 0);
        ASSERT_EQ(m.runLen(id), 2);
        ASSERT_TRUE(m.setContains(k, k));
        ASSERT_TRUE(m.setContains(k, k + 100));
    }
}

TEST(cold_hash_map, set_map_deload_byte_identity_facets) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSetMap a(&lb, &dirty);
    // 30 keys, key k carrying a sorted set of (k % 4) + 1 values built out of
    // order (insertSorted keeps each run canonical).
    for (int32_t k = 0; k < 30; ++k) {
        const int32_t n = (k % 4) + 1;
        for (int32_t j = n - 1; j >= 0; --j) a.insertSorted(200 + k, k * 10 + j);
    }

    // Dump the three tags through the facets (the LbMemory deload path).
    PodSetMap::KeysView aKeys(&a);
    PodSetMap::RunStartsView aRuns(&a);
    PodSetMap::RunValuesView aVals(&a);
    std::vector<char> kb, rb, vb;
    aKeys.appendSpanBytes(kb, 0);
    aRuns.appendSpanBytes(rb, 0);
    aVals.appendSpanBytes(vb, 0);
    ASSERT_EQ(aKeys.size(), 30);
    ASSERT_EQ(aRuns.size(), 30);             // one run-start per key
    ASSERT_EQ(aVals.size(), a.valueCount()); // value total, not key count

    // Reload onto a fresh arena through the facets, key tag first.
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodSetMap b(&lb2, &dirty2);
    PodSetMap::KeysView(&b).bulkAppendBytes(kb.data(), aKeys.size());
    PodSetMap::RunStartsView(&b).bulkAppendBytes(rb.data(), aRuns.size());
    PodSetMap::RunValuesView(&b).bulkAppendBytes(vb.data(), aVals.size());

    ASSERT_EQ(b.count(), a.count());
    ASSERT_EQ(b.valueCount(), a.valueCount());
    for (int32_t id = 1; id <= a.count(); ++id) {
        ASSERT_EQ(b.runLen(id), a.runLen(id));
        for (int32_t j = 0; j < a.runLen(id); ++j)
            ASSERT_EQ(b.valueAt(id, j), a.valueAt(id, j));
        ASSERT_EQ(b.lookup(b.keyAt(id)), a.lookup(a.keyAt(id)));
    }
    std::vector<char> kb2, rb2, vb2;
    PodSetMap::KeysView(&b).appendSpanBytes(kb2, 0);
    PodSetMap::RunStartsView(&b).appendSpanBytes(rb2, 0);
    PodSetMap::RunValuesView(&b).appendSpanBytes(vb2, 0);
    ASSERT_TRUE(kb2 == kb);
    ASSERT_TRUE(rb2 == rb);
    ASSERT_TRUE(vb2 == vb);
}

TEST(cold_hash_map, set_map_erase_if_compaction_byte_identical) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);

    // A set-map whose key + run + value columns span several pages.
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodSetMap m(&lb, &d);
    const int32_t N = 1500;
    for (int32_t k = 0; k < N; ++k) {
        m.insertSorted(k, k);
        m.insertSorted(k, k + 1000000);
    }

    // Erase every key divisible by 3 (scattered across pages) in one pass.
    const int32_t removed = m.eraseSetIf([](int32_t k) { return k % 3 == 0; });
    int32_t expectRemoved = 0;
    for (int32_t k = 0; k < N; ++k) if (k % 3 == 0) ++expectRemoved;
    ASSERT_EQ(removed, expectRemoved);
    ASSERT_EQ(m.count(), N - expectRemoved);

    // The compacted columns are byte-identical to a from-scratch insert of the
    // survivors in order (the canonical-bytes guarantee of compacting erase).
    gl::LbArena lb2(&g);
    gl::DirtyState d2 = gl::DirtyState::Clean;
    PodSetMap fresh(&lb2, &d2);
    for (int32_t k = 0; k < N; ++k) {
        if (k % 3 == 0) continue;
        fresh.insertSorted(k, k);
        fresh.insertSorted(k, k + 1000000);
    }
    std::vector<char> mk, mr, mv, fk, fr, fv;
    m.keyStore().appendKeyBytes(mk, 0);
    m.appendRunStartBytes(mr, 0);
    m.appendValueBytes(mv, 0);
    fresh.keyStore().appendKeyBytes(fk, 0);
    fresh.appendRunStartBytes(fr, 0);
    fresh.appendValueBytes(fv, 0);
    ASSERT_TRUE(mk == fk);
    ASSERT_TRUE(mr == fr);
    ASSERT_TRUE(mv == fv);
}

TEST(cold_hash_map, set_map_copy_from_deep_copy) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSetMap src(&lb, &dirty);
    for (int32_t k = 0; k < 20; ++k) {
        src.insertSorted(k, k * 2);
        src.insertSorted(k, k * 2 + 1);
    }

    gl::LbArena lbClone(&g);
    gl::DirtyState dirtyClone = gl::DirtyState::Clean;
    PodSetMap clone(&lbClone, &dirtyClone);
    clone.copyFrom(src);
    ASSERT_EQ(clone.count(), src.count());
    ASSERT_EQ(clone.valueCount(), src.valueCount());
    for (int32_t k = 0; k < 20; ++k) {
        ASSERT_TRUE(clone.setContains(k, k * 2));
        ASSERT_TRUE(clone.setContains(k, k * 2 + 1));
    }
    // The clone mutates independently; the source is unchanged.
    clone.insertSorted(999, 7);
    ASSERT_TRUE(clone.setContains(999, 7));
    ASSERT_EQ(src.lookup(999), 0);
}

TEST(cold_hash_map, bytes_key_map_erase_canonical) {
    // Byte-key single-value map erase (the eqClassSttmntIndexMapMap shape after
    // flattening to (validity ++ memberIds) -> int). The compaction slides only
    // the survivor LOCATION entries; the erased keys' bytes become holes in the
    // append-only byte pool, yet the canonical deload stream (per-key logical
    // bytes in id order) stays byte-identical to a dense from-scratch insert.
    using ByteMap = gl::ColdHashMap<gl::BytesKeyStore, int32_t>;
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);

    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    ByteMap m(&lb, &d);
    std::vector<std::string> keys;
    for (int32_t i = 0; i < 40; ++i)
        keys.push_back("(k" + std::to_string(i) + ","
                       + std::string(static_cast<size_t>((i % 6) + 1), 'a') + ")");
    for (int32_t i = 0; i < 40; ++i) m.insert(gl::StrSpan(keys[i]), 1000 + i);
    int64_t originalKeyBytes = 0;
    for (const std::string& key : keys)
        originalKeyBytes += static_cast<int64_t>(key.size());
    ASSERT_EQ(m.keyStore().logicalByteCount(), originalKeyBytes);
    ASSERT_EQ(m.keyStore().contentBytesFrom(0), originalKeyBytes);

    // Single-key erase (interior) + a content-predicate bulk eraseIf.
    ASSERT_TRUE(m.erase(gl::StrSpan(keys[7])));
    ASSERT_FALSE(m.erase(gl::StrSpan(keys[7])));          // already gone
    ASSERT_TRUE(m.find(gl::StrSpan(keys[7])) == nullptr);
    const int32_t removed =
        m.eraseIf([](const gl::StrSpan& s) { return s.len >= 9; });
    ASSERT_TRUE(removed > 0);

    auto survives = [&](int32_t i) {
        return i != 7 && static_cast<int32_t>(keys[i].size()) < 9;
    };
    for (int32_t i = 0; i < 40; ++i) {
        const int32_t* v = m.find(gl::StrSpan(keys[i]));
        if (survives(i)) ASSERT_TRUE(v != nullptr && *v == 1000 + i);
        else ASSERT_TRUE(v == nullptr);
    }
    int64_t survivorKeyBytes = 0;
    for (int32_t i = 0; i < 40; ++i)
        if (survives(i))
            survivorKeyBytes += static_cast<int64_t>(keys[i].size());
    ASSERT_EQ(m.keyStore().logicalByteCount(), survivorKeyBytes);
    ASSERT_EQ(m.keyStore().contentBytesFrom(0), survivorKeyBytes);

    // Deload stream is byte-identical to a fresh dense insert of the survivors.
    gl::LbArena lb2(&g);
    gl::DirtyState d2 = gl::DirtyState::Clean;
    ByteMap fresh(&lb2, &d2);
    for (int32_t i = 0; i < 40; ++i)
        if (survives(i)) fresh.insert(gl::StrSpan(keys[i]), 1000 + i);
    ASSERT_EQ(m.count(), fresh.count());

    std::vector<char> ml, mb, mv, fl, fb, fv;
    ByteMap::LengthsView(&m).appendSpanBytes(ml, 0);
    ByteMap::BytesView(&m).appendSpanBytes(mb, 0);
    m.appendValueBytes(mv, 0);
    ByteMap::LengthsView(&fresh).appendSpanBytes(fl, 0);
    ByteMap::BytesView(&fresh).appendSpanBytes(fb, 0);
    fresh.appendValueBytes(fv, 0);
    ASSERT_TRUE(ml == fl);
    ASSERT_TRUE(mb == fb);
    ASSERT_TRUE(mv == fv);

    // Reload the compacted map through the facets (holes gone, dense pool).
    gl::LbArena lb3(&g);
    gl::DirtyState d3 = gl::DirtyState::Clean;
    ByteMap r(&lb3, &d3);
    ByteMap::LengthsView(&r).bulkAppendBytes(ml.data(), ByteMap::LengthsView(&m).size());
    ByteMap::BytesView(&r).bulkAppendBytes(mb.data(), ByteMap::BytesView(&m).size());
    ByteMap::ValuesView(&r).bulkAppendBytes(mv.data(), ByteMap::ValuesView(&m).size());
    ASSERT_EQ(r.count(), m.count());
    ASSERT_EQ(r.keyStore().logicalByteCount(), survivorKeyBytes);
    for (int32_t i = 0; i < 40; ++i)
        if (survives(i)) {
            const int32_t* v = r.find(gl::StrSpan(keys[i]));
            ASSERT_TRUE(v != nullptr && *v == 1000 + i);
        }
}

// ---- ColdBlobMap — the record value store ------------------------------

TEST(cold_hash_map, blob_map_alias_identity) {
    static_assert(std::is_same<PodBlobMap,
                      gl::HashMap<gl::PodKeyStore<int32_t>,
                                  gl::BlobCsrValueStore>>::value,
                  "ColdBlobMap must alias HashMap<KeyStore, BlobCsrValueStore>");
    static_assert(gl::BlobCsrValueStore::kTagCount == 3,
                  "blob value store contributes three tags");
    ASSERT_TRUE(true);
}

TEST(cold_hash_map, blob_run_start_suffix_rebase_matches_scalar_oracle) {
    gl::GlobalMemoryManager manager;
    manager.init(kMapTestCfg);
    gl::LbArena arena(&manager);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::BlobCsrValueStore store(&arena, &dirty);
    const char byte = 'x';
    constexpr int32_t kRuns = 5000;
    for (int32_t i = 0; i < kRuns; ++i) {
        store.openRun();
        store.appendBlob(&byte, 1);
    }

    dirty = gl::DirtyState::Clean;
    constexpr int32_t kFirst = 1023;
    constexpr int32_t kDelta = 37;
    store.addToRunStartsSuffix(kFirst, kDelta);

    ASSERT_EQ(static_cast<int>(dirty),
              static_cast<int>(gl::DirtyState::Restructured));
    for (int32_t i = 0; i < kRuns; ++i) {
        const int32_t expected = i + ((i >= kFirst) ? kDelta : 0);
        ASSERT_EQ(store.runStartRaw(i), expected);
    }
}

TEST(cold_hash_map, blob_value_store_generated_replace_matches_contiguous) {
    gl::GlobalMemoryManager manager;
    manager.init(kMapTestCfg);
    gl::LbArena arena(&manager);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::BlobCsrValueStore store(&arena, &dirty);
    store.openRun();
    store.appendBlob("old-a", 5);
    store.appendBlob("old-bb", 6);
    store.openRun();
    store.appendBlob("tail", 4);

    const std::vector<std::string> replacement = {
        "generated-one", "g2", "generated-three"
    };
    int32_t total = 0;
    for (const std::string& value : replacement)
        total += static_cast<int32_t>(value.size());
    store.replaceRunGenerated(0, 2,
        static_cast<int32_t>(replacement.size()), total,
        [&](const auto& sink) {
            for (const std::string& value : replacement)
                sink(value.data(), static_cast<int32_t>(value.size()));
        });
    store.addToRunStartsSuffix(1, 1);

    ASSERT_EQ(store.runLen(1, 2), 3);
    ASSERT_EQ(store.runLen(2, 2), 1);
    std::vector<char> bytes;
    for (int32_t j = 0; j < 3; ++j) {
        store.readBlob(j, bytes);
        ASSERT_TRUE(std::string(bytes.begin(), bytes.end()) == replacement[j]);
    }
    store.readBlob(3, bytes);
    ASSERT_TRUE(std::string(bytes.begin(), bytes.end()) == "tail");
}

TEST(cold_hash_map, blob_map_assign_run_generated_matches_contiguous) {
    gl::GlobalMemoryManager manager;
    manager.init(kMapTestCfg);
    gl::LbArena arenaA(&manager), arenaB(&manager);
    gl::DirtyState dirtyA = gl::DirtyState::Clean;
    gl::DirtyState dirtyB = gl::DirtyState::Clean;
    PodBlobMap generated(&arenaA, &dirtyA);
    PodBlobMap contiguous(&arenaB, &dirtyB);
    assignBlobs(generated, 10, { "old-a", "old-b" });
    assignBlobs(generated, 20, { "tail-a", "tail-b" });
    assignBlobs(contiguous, 10, { "old-a", "old-b" });
    assignBlobs(contiguous, 20, { "tail-a", "tail-b" });

    const std::vector<std::string> replacement = {
        "segmented-alpha", "b", "segmented-gamma", "delta"
    };
    std::string concat;
    std::vector<int32_t> lens;
    for (const std::string& value : replacement) {
        concat += value;
        lens.push_back(static_cast<int32_t>(value.size()));
    }
    generated.assignRunGenerated(10,
        static_cast<int32_t>(replacement.size()),
        static_cast<int32_t>(concat.size()), [&](const auto& sink) {
            for (const std::string& value : replacement)
                sink(value.data(), static_cast<int32_t>(value.size()));
        });
    contiguous.assignRun(10, concat.data(), lens.data(),
                         static_cast<int32_t>(lens.size()));
    generated.assignRunGenerated(30, 2, 9, [&](const auto& sink) {
        sink("new", 3);
        sink("record", 6);
    });
    const int32_t newLens[2] = { 3, 6 };
    contiguous.assignRun(30, "newrecord", newLens, 2);

    ASSERT_EQ(generated.count(), contiguous.count());
    ASSERT_EQ(generated.blobCount(), contiguous.blobCount());
    for (int32_t id = 1; id <= generated.count(); ++id) {
        ASSERT_EQ(generated.runLen(id), contiguous.runLen(id));
        for (int32_t j = 0; j < generated.runLen(id); ++j)
            ASSERT_TRUE(readBlobStr(generated, id, j)
                == readBlobStr(contiguous, id, j));
    }
}

// A run whose TOTAL bytes exceed one pool block, written via the per-blob
// emitter door (assignRunGenerated) from individually staged blobs, must be
// byte-identical to the contiguous-concat assignRun reference — the eq-class
// run-splice conversion's contract (each single blob stays <= one block; only
// the whole-run concat ceiling is removed).
TEST(cold_hash_map, blob_map_generated_multi_block_run_matches_contiguous) {
    gl::GlobalMemoryManager manager;
    manager.init(gl::StaticMemoryConfig{ 4 << 20, 1 << 18 });   // 4 MiB / 256 KiB
    gl::LbArena arenaA(&manager), arenaB(&manager);
    gl::DirtyState dirtyA = gl::DirtyState::Clean;
    gl::DirtyState dirtyB = gl::DirtyState::Clean;
    PodBlobMap generated(&arenaA, &dirtyA);
    PodBlobMap contiguous(&arenaB, &dirtyB);

    const auto makeRun = [](int32_t blobCount, int32_t blobLen, char salt) {
        std::vector<std::string> run;
        run.reserve(static_cast<std::size_t>(blobCount));
        for (int32_t i = 0; i < blobCount; ++i) {
            std::string b(static_cast<std::size_t>(blobLen), ' ');
            for (int32_t p = 0; p < blobLen; ++p)
                b[static_cast<std::size_t>(p)] =
                    static_cast<char>('a' + ((i * 31 + p * 7 + salt) % 23));
            run.push_back(std::move(b));
        }
        return run;
    };
    const auto writeBoth = [&](int32_t key, const std::vector<std::string>& run) {
        std::string concat;
        std::vector<int32_t> lens;
        int64_t total = 0;
        for (const std::string& b : run) {
            concat += b;
            lens.push_back(static_cast<int32_t>(b.size()));
            total += static_cast<int64_t>(b.size());
        }
        ASSERT_TRUE(total > (1 << 18));               // the run spans > one block
        generated.assignRunGenerated(key,
            static_cast<int32_t>(run.size()),
            static_cast<int32_t>(total), [&](const auto& sink) {
                for (const std::string& b : run)
                    sink(b.data(), static_cast<int32_t>(b.size()));
            });
        contiguous.assignRun(key, concat.data(), lens.data(),
                             static_cast<int32_t>(lens.size()));
    };

    // Fresh key: 40 x 8000 B = 320 KB run.
    writeBoth(10, makeRun(40, 8000, 1));
    // Existing-key replace: a different >block run through replaceRunGenerated.
    writeBoth(10, makeRun(45, 7000, 2));

    ASSERT_EQ(generated.count(), contiguous.count());
    ASSERT_EQ(generated.blobCount(), contiguous.blobCount());
    for (int32_t id = 1; id <= generated.count(); ++id) {
        ASSERT_EQ(generated.runLen(id), contiguous.runLen(id));
        for (int32_t j = 0; j < generated.runLen(id); ++j)
            ASSERT_TRUE(readBlobStr(generated, id, j)
                == readBlobStr(contiguous, id, j));
    }
}

TEST(cold_hash_map, blob_map_known_id_assignment_matches_key_probe) {
    gl::GlobalMemoryManager manager;
    manager.init(kMapTestCfg);
    gl::LbArena arenaA(&manager), arenaB(&manager);
    gl::DirtyState dirtyA = gl::DirtyState::Clean;
    gl::DirtyState dirtyB = gl::DirtyState::Clean;
    PodBlobMap knownId(&arenaA, &dirtyA);
    PodBlobMap keyProbe(&arenaB, &dirtyB);
    assignBlobs(knownId, 10, { "old", "records" });
    assignBlobs(knownId, 20, { "tail" });
    assignBlobs(keyProbe, 10, { "old", "records" });
    assignBlobs(keyProbe, 20, { "tail" });

    const BlobBatch contiguous = makeBlobs({ "alpha", "b", "gamma" });
    ASSERT_EQ(knownId.assignRunAtId(1, contiguous.bytes.data(),
                                   contiguous.lens.data(), 3), 1);
    ASSERT_EQ(keyProbe.assignRun(10, contiguous.bytes.data(),
                                 contiguous.lens.data(), 3), 1);

    ASSERT_EQ(knownId.assignRunGeneratedAtId(1, 2, 11,
        [&](const auto& sink) {
            sink("generated", 9);
            sink("id", 2);
        }), 1);
    ASSERT_EQ(keyProbe.assignRunGenerated(10, 2, 11,
        [&](const auto& sink) {
            sink("generated", 9);
            sink("id", 2);
        }), 1);

    ASSERT_EQ(knownId.count(), keyProbe.count());
    ASSERT_EQ(knownId.blobCount(), keyProbe.blobCount());
    for (int32_t id = 1; id <= knownId.count(); ++id) {
        ASSERT_EQ(knownId.runLen(id), keyProbe.runLen(id));
        for (int32_t j = 0; j < knownId.runLen(id); ++j)
            ASSERT_TRUE(readBlobStr(knownId, id, j)
                == readBlobStr(keyProbe, id, j));
    }
}

TEST(cold_hash_map, blob_range_scanner_matches_blob_reads_across_pages) {
    gl::GlobalMemoryManager manager;
    manager.init(kMapTestCfg);
    gl::LbArena arena(&manager), scratch(&manager);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::BlobCsrValueStore store(&arena, &dirty);
    const std::vector<std::string> expected = {
        std::string(5000, 'a'), std::string(5000, 'b'),
        std::string(), std::string(7000, 'c')
    };
    store.openRun();
    for (const std::string& value : expected)
        store.appendBlob(value.data(), static_cast<int32_t>(value.size()));

    std::vector<std::string> scanned;
    store.forEachBlobRange(0, static_cast<int32_t>(expected.size()), scratch,
        [&](const char* bytes, int32_t len) {
            scanned.emplace_back(bytes == nullptr ? "" : std::string(bytes, len));
        });
    ASSERT_TRUE(scanned == expected);
}

TEST(cold_hash_map, blob_map_run_scanner_matches_random_access) {
    gl::GlobalMemoryManager manager;
    manager.init(kMapTestCfg);
    gl::LbArena arena(&manager), scratch(&manager);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodBlobMap map(&arena, &dirty);
    const std::vector<std::string> expected = {
        std::string(5000, 'x'), std::string(5000, 'y'),
        std::string(7000, 'z')
    };
    const int32_t id = assignBlobs(map, 77, expected);
    assignBlobs(map, 88, { "later-key" });

    std::vector<std::string> scanned;
    map.forEachBlobContiguous(id, scratch,
        [&](const char* bytes, int32_t len) {
            scanned.emplace_back(bytes, len);
        });
    ASSERT_TRUE(scanned == expected);
    for (int32_t j = 0; j < static_cast<int32_t>(expected.size()); ++j)
        ASSERT_TRUE(readBlobStr(map, id, j) == scanned[j]);
}

// `PagedVector::replaceRange` against a `std::vector` oracle, across many pages.
TEST(paged_vector, replace_range_matches_std_vector_oracle) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::PagedVector<int32_t> v(&lb, &d);
    std::vector<int32_t> oracle;
    const int32_t N = 5000;                    // spans many pages
    for (int32_t i = 0; i < N; ++i) { v.push_back(i); oracle.push_back(i); }

    auto apply = [&](int32_t pos, int32_t oldLen, std::vector<int32_t> repl) {
        v.replaceRange(pos, oldLen, repl.data(),
                       static_cast<int32_t>(repl.size()));
        oracle.erase(oracle.begin() + pos, oracle.begin() + pos + oldLen);
        oracle.insert(oracle.begin() + pos, repl.begin(), repl.end());
    };
    apply(100, 3, { -1, -2, -3 });                              // same size
    apply(200, 2, { 1000, 1001, 1002, 1003, 1004, 1005, 1006 }); // grow
    apply(400, 9, { 7 });                                       // shrink
    apply(0, 10, {});                                           // erase range
    apply(50, 0, { 8888, 9999 });                               // insert range
    apply(static_cast<int32_t>(oracle.size()) - 1, 1, { 5, 6, 7 }); // tail grow
    apply(static_cast<int32_t>(oracle.size()), 0, { 42 });      // append

    ASSERT_EQ(v.size(), static_cast<int32_t>(oracle.size()));
    for (int32_t i = 0; i < static_cast<int32_t>(oracle.size()); ++i)
        ASSERT_EQ(v[i], oracle[i]);
}

TEST(cold_hash_map, blob_map_basic_runs) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodBlobMap m(&lb, &d);

    ASSERT_EQ(assignBlobs(m, 10, { "alpha", "be", "gamma!" }), 1);
    ASSERT_EQ(assignBlobs(m, 20, { "x" }), 2);
    ASSERT_EQ(assignBlobs(m, 30, {}), 3);                 // empty run
    ASSERT_EQ(assignBlobs(m, 40, { "", "zz" }), 4);       // empty blob, then one

    ASSERT_EQ(m.count(), 4);
    ASSERT_EQ(m.blobCount(), 3 + 1 + 0 + 2);
    ASSERT_EQ(m.runLen(1), 3);
    ASSERT_EQ(m.runLen(3), 0);
    ASSERT_EQ(readBlobStr(m, 1, 0), "alpha");
    ASSERT_EQ(readBlobStr(m, 1, 2), "gamma!");
    ASSERT_EQ(readBlobStr(m, 2, 0), "x");
    ASSERT_EQ(readBlobStr(m, 4, 0), "");
    ASSERT_EQ(readBlobStr(m, 4, 1), "zz");
    ASSERT_EQ(m.lookup(30), 3);
    ASSERT_EQ(m.lookup(999), 0);
}

TEST(cold_hash_map, blob_map_assign_run_grow_shrink_same) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodBlobMap m(&lb, &d);

    assignBlobs(m, 100, { "aa", "bbbb", "cc" });   // id 1
    assignBlobs(m, 200, { "tail1", "tail2" });     // id 2 (trailing key)

    // Same total bytes, different split: interior overwrite, no shift.
    assignBlobs(m, 100, { "xx", "yyyy", "zz" });
    ASSERT_EQ(m.runLen(1), 3);
    ASSERT_EQ(readBlobStr(m, 1, 1), "yyyy");
    ASSERT_EQ(readBlobStr(m, 2, 0), "tail1");      // tail aligned
    ASSERT_EQ(readBlobStr(m, 2, 1), "tail2");

    // Grow (more blobs + more bytes): shifts the tail key right.
    assignBlobs(m, 100, { "0", "11", "222", "3333", "44444" });
    ASSERT_EQ(m.runLen(1), 5);
    ASSERT_EQ(readBlobStr(m, 1, 4), "44444");
    ASSERT_EQ(readBlobStr(m, 2, 0), "tail1");
    ASSERT_EQ(readBlobStr(m, 2, 1), "tail2");

    // Shrink to a single small blob: shifts the tail key back.
    assignBlobs(m, 100, { "q" });
    ASSERT_EQ(m.runLen(1), 1);
    ASSERT_EQ(readBlobStr(m, 1, 0), "q");
    ASSERT_EQ(m.runLen(2), 2);
    ASSERT_EQ(readBlobStr(m, 2, 0), "tail1");

    // Shrink to an empty run.
    assignBlobs(m, 100, {});
    ASSERT_EQ(m.runLen(1), 0);
    ASSERT_EQ(readBlobStr(m, 2, 1), "tail2");
}

TEST(cold_hash_map, blob_map_append_blob_to_run) {
    // appendBlobToRun(id, blob, len) appends ONE blob at key id's run-end without
    // touching the existing blobs, rebasing later keys' run starts -- the
    // encodedMap RMW fast path. The result must equal a one-shot assignRun of the
    // full run, byte-for-byte, including when the grown key is NOT the pool tail.
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);

    gl::LbArena lbA(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    PodBlobMap viaAppend(&lbA, &dA);
    assignBlobs(viaAppend, 100, { "aa", "bbbb" });      // id 1 (non-tail)
    assignBlobs(viaAppend, 200, { "tail1", "tail2" });  // id 2 (tail)
    const auto append1 = [&](int32_t id, const std::string& s) {
        viaAppend.appendBlobToRun(id, s.data(), static_cast<int32_t>(s.size()));
    };
    append1(1, "cc");      // grow the NON-tail key -> shifts key 2 right
    append1(2, "tail3");   // grow the tail key -> pure pool-tail append

    gl::LbArena lbB(&g);
    gl::DirtyState dB = gl::DirtyState::Clean;
    PodBlobMap viaAssign(&lbB, &dB);
    assignBlobs(viaAssign, 100, { "aa", "bbbb", "cc" });
    assignBlobs(viaAssign, 200, { "tail1", "tail2", "tail3" });

    // Runs grew in place; later key intact; appended blob at the run-end.
    ASSERT_EQ(viaAppend.runLen(1), 3);
    ASSERT_EQ(viaAppend.runLen(2), 3);
    ASSERT_EQ(readBlobStr(viaAppend, 1, 0), "aa");
    ASSERT_EQ(readBlobStr(viaAppend, 1, 2), "cc");
    ASSERT_EQ(readBlobStr(viaAppend, 2, 0), "tail1");
    ASSERT_EQ(readBlobStr(viaAppend, 2, 2), "tail3");

    // Byte-identity of all three value columns vs the one-shot assignRun build.
    std::vector<char> ra, ba, pa, rb, bb, pb;
    viaAppend.appendRunStartBytes(ra, 0);
    viaAppend.appendBlobStartBytes(ba, 0);
    viaAppend.appendBlobPoolBytes(pa, 0);
    viaAssign.appendRunStartBytes(rb, 0);
    viaAssign.appendBlobStartBytes(bb, 0);
    viaAssign.appendBlobPoolBytes(pb, 0);
    ASSERT_TRUE(ra == rb);
    ASSERT_TRUE(ba == bb);
    ASSERT_TRUE(pa == pb);
}

TEST(cold_hash_map, blob_map_erase_blob_from_run) {
    // eraseBlobFromRun(id, j) removes ONE blob of key id's run without
    // rewriting the others and rebases later keys' run starts -- the owner-
    // removal fast path. Every erase must equal re-assigning the run without
    // that blob, column for column: first, middle and last blob, an empty
    // blob, a non-tail key, the pool's last blob, and runs emptied to zero
    // blobs (the key keeps its id).
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lbA(&g), lbB(&g);
    gl::DirtyState dA = gl::DirtyState::Clean, dB = gl::DirtyState::Clean;
    PodBlobMap viaErase(&lbA, &dA), viaAssign(&lbB, &dB);
    std::vector<std::vector<std::string>> runs = {
        { "aa", "bbbb", "cc", "" }, { "tail1", "tail2", "tail3" }, { "solo" } };
    for (int32_t k = 0; k < 3; ++k) {
        assignBlobs(viaErase, 100 * (k + 1), runs[k]);
        assignBlobs(viaAssign, 100 * (k + 1), runs[k]);
    }
    const auto sameColumns = [&]() {
        std::vector<char> x, y;
        viaErase.appendRunStartBytes(x, 0); viaAssign.appendRunStartBytes(y, 0); ASSERT_TRUE(x == y);
        x.clear(); y.clear();
        viaErase.appendBlobStartBytes(x, 0); viaAssign.appendBlobStartBytes(y, 0); ASSERT_TRUE(x == y);
        x.clear(); y.clear();
        viaErase.appendBlobPoolBytes(x, 0); viaAssign.appendBlobPoolBytes(y, 0); ASSERT_TRUE(x == y);
        ASSERT_EQ(viaErase.count(), 3);
        for (int32_t id = 1; id <= 3; ++id) {
            ASSERT_EQ(viaErase.runLen(id), static_cast<int32_t>(runs[id - 1].size()));
            for (int32_t j = 0; j < viaErase.runLen(id); ++j)
                ASSERT_EQ(readBlobStr(viaErase, id, j), runs[id - 1][j]);
        }
    };
    const auto erase = [&](int32_t id, int32_t j) {
        viaErase.eraseBlobFromRun(id, j);
        runs[id - 1].erase(runs[id - 1].begin() + j);
        assignBlobs(viaAssign, 100 * id, runs[id - 1]);
        sameColumns();
    };
    erase(1, 1);   // middle blob of a non-tail key
    erase(1, 2);   // its empty last blob
    erase(1, 0);   // its first blob
    erase(2, 2);   // the pool's last blob
    erase(3, 0);   // the tail key's run emptied
    erase(2, 0);
    erase(1, 0);   // a non-tail run emptied
    erase(2, 0);   // every run empty
}

TEST(cold_hash_map, shared_blob_map_erase_blob_from_run_keeps_identifiers) {
    // In a store bound to a shared byte pool (D-440) the erase splices out
    // one stored identifier: the survivors keep theirs, the pool gains no
    // payload, and the logical columns equal a private oracle re-assigned
    // without the blob. A private oversized survivor keeps its handle and
    // reads back intact; an erased oversized record leaves the resident pool
    // at the next quiescent collection.
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{16 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), otherArena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::SharedBytePool pool(&arena, &dirty);
    PodBlobMap shared(&arena, &dirty), oracle(&otherArena, &dirty);
    shared.bindSharedBlobBytes(pool, true);
    const std::string a(700, 'a'), b(800, 'b'), c(900, 'c'), large(12000, 'l');
    std::vector<std::vector<std::string>> runs = { { a, b, large, c }, { c, a } };
    for (int32_t k = 0; k < 2; ++k) {
        assignBlobs(shared, k + 1, runs[k]);
        assignBlobs(oracle, k + 1, runs[k]);
    }
    ASSERT_EQ(pool.count(), 3);
    const auto same = [&]() {
        std::vector<char> x, y;
        shared.appendRunStartBytes(x, 0); oracle.appendRunStartBytes(y, 0); ASSERT_TRUE(x == y);
        x.clear(); y.clear();
        shared.appendBlobStartBytes(x, 0); oracle.appendBlobStartBytes(y, 0); ASSERT_TRUE(x == y);
        x.clear(); y.clear();
        shared.appendBlobPoolBytes(x, 0); oracle.appendBlobPoolBytes(y, 0); ASSERT_TRUE(x == y);
        ASSERT_EQ(shared.poolByteCount(), oracle.poolByteCount());
        for (int32_t id = 1; id <= 2; ++id) {
            ASSERT_EQ(shared.runLen(id), static_cast<int32_t>(runs[id - 1].size()));
            for (int32_t j = 0; j < shared.runLen(id); ++j)
                ASSERT_TRUE(readBlobStr(shared, id, j) == runs[id - 1][j]);
        }
    };
    const auto erase = [&](int32_t id, int32_t j) {
        shared.eraseBlobFromRun(id, j);
        runs[id - 1].erase(runs[id - 1].begin() + j);
        assignBlobs(oracle, id, runs[id - 1]);
        same();
        ASSERT_EQ(pool.count(), 3);   // nothing minted, nothing dropped before a collection
    };
    erase(1, 1);   // b goes; the oversized survivor shifts left with its handle
    ASSERT_EQ(shared.residentPoolByteCount(), shared.blobCount() * 4 + 12000);
    erase(2, 0);   // c leaves run 2 but stays referenced from run 1
    erase(1, 1);   // the oversized record goes
    pool.collect(scratch); same();
    ASSERT_EQ(shared.residentPoolByteCount(), shared.blobCount() * 4);
    ASSERT_EQ(pool.count(), 2);   // a and c remain referenced
    shared.release(); pool.release();
    ASSERT_EQ(pool.count(), 0);
}

TEST(cold_hash_map, erase_blob_from_run_assertion_probe) {
    // Negative coverage, driven by .scripts/tests/test_native_run_erase.py:
    // erasing outside a key's run or on a non-existent key asserts.
    std::string mode;
#ifdef _WIN32
    char* value = nullptr; size_t length = 0;
    ASSERT_EQ(_dupenv_s(&value, &length, "GL_TEST_INVALID_RUN_ERASE"), 0);
    if (value != nullptr) { mode = value; std::free(value); }
#else
    if (const char* value = std::getenv("GL_TEST_INVALID_RUN_ERASE")) mode = value;
#endif
    gl::GlobalMemoryManager global; global.init(kMapTestCfg);
    gl::LbArena arena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodBlobMap values(&arena, &dirty);
    assignBlobs(values, 1, {"only"});
    if (mode == "outside_run") values.eraseBlobFromRun(1, 1);
    else if (mode == "missing_key") values.eraseBlobFromRun(2, 0);
}

TEST(cold_hash_map, blob_map_erase_blob_if) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodBlobMap m(&lb, &d);
    for (int32_t k = 0; k < 10; ++k)
        assignBlobs(m, k,
            { "k" + std::to_string(k), "v" + std::to_string(k * 7) });
    ASSERT_EQ(m.count(), 10);

    // No-match eraseBlobIf is a pure no-op (no dirty escalation).
    d = gl::DirtyState::Clean;
    ASSERT_EQ(m.eraseBlobIf([](int32_t k) { return k > 1000; }), 0);
    ASSERT_EQ(static_cast<int>(d), static_cast<int>(gl::DirtyState::Clean));

    // Drop the evens in one pass; survivors keep their runs + bytes aligned.
    const int32_t removed = m.eraseBlobIf([](int32_t k) { return (k % 2) == 0; });
    ASSERT_EQ(removed, 5);
    ASSERT_EQ(m.count(), 5);
    for (int32_t k = 1; k < 10; k += 2) {
        const int32_t id = m.lookup(k);
        ASSERT_TRUE(id != 0);
        ASSERT_EQ(m.runLen(id), 2);
        ASSERT_EQ(readBlobStr(m, id, 0), "k" + std::to_string(k));
        ASSERT_EQ(readBlobStr(m, id, 1), "v" + std::to_string(k * 7));
    }
    for (int32_t k = 0; k < 10; k += 2) ASSERT_EQ(m.lookup(k), 0);
}

TEST(cold_hash_map, blob_map_deload_byte_identity_facets) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodBlobMap a(&lb, &d);
    for (int32_t k = 0; k < 40; ++k) {
        std::vector<std::string> blobs;
        const int32_t n = (k % 3) + (k % 2);          // 0..3 blobs (empty runs too)
        for (int32_t j = 0; j < n; ++j)
            blobs.push_back(
                std::string(static_cast<size_t>((k + j) % 5),
                            static_cast<char>('a' + (j % 7)))
                + std::to_string(k));
        assignBlobs(a, 500 + k, blobs);
    }

    PodBlobMap::KeysView aKeys(&a);
    PodBlobMap::RunStartsView aRuns(&a);
    PodBlobMap::BlobStartsView aBlobs(&a);
    PodBlobMap::BlobPoolView aPool(&a);
    std::vector<char> kb, rb, bb, pb;
    aKeys.appendSpanBytes(kb, 0);
    aRuns.appendSpanBytes(rb, 0);
    aBlobs.appendSpanBytes(bb, 0);
    aPool.appendSpanBytes(pb, 0);
    ASSERT_EQ(aKeys.size(), 40);
    ASSERT_EQ(aRuns.size(), 40);                       // one run-start per key
    ASSERT_EQ(aBlobs.size(), a.blobCount());
    ASSERT_EQ(aPool.size(), a.poolByteCount());

    // Reload onto a fresh arena through the facets, key tag first.
    gl::LbArena lb2(&g);
    gl::DirtyState d2 = gl::DirtyState::Clean;
    PodBlobMap b(&lb2, &d2);
    PodBlobMap::KeysView(&b).bulkAppendBytes(kb.data(), aKeys.size());
    PodBlobMap::RunStartsView(&b).bulkAppendBytes(rb.data(), aRuns.size());
    PodBlobMap::BlobStartsView(&b).bulkAppendBytes(bb.data(), aBlobs.size());
    PodBlobMap::BlobPoolView(&b).bulkAppendBytes(pb.data(), aPool.size());

    ASSERT_EQ(b.count(), a.count());
    ASSERT_EQ(b.blobCount(), a.blobCount());
    ASSERT_EQ(b.poolByteCount(), a.poolByteCount());
    for (int32_t id = 1; id <= a.count(); ++id) {
        ASSERT_EQ(b.runLen(id), a.runLen(id));
        for (int32_t j = 0; j < a.runLen(id); ++j)
            ASSERT_EQ(readBlobStr(b, id, j), readBlobStr(a, id, j));
        ASSERT_EQ(b.lookup(b.keyAt(id)), a.lookup(a.keyAt(id)));
    }
    std::vector<char> kb2, rb2, bb2, pb2;
    PodBlobMap::KeysView(&b).appendSpanBytes(kb2, 0);
    PodBlobMap::RunStartsView(&b).appendSpanBytes(rb2, 0);
    PodBlobMap::BlobStartsView(&b).appendSpanBytes(bb2, 0);
    PodBlobMap::BlobPoolView(&b).appendSpanBytes(pb2, 0);
    ASSERT_TRUE(kb2 == kb);
    ASSERT_TRUE(rb2 == rb);
    ASSERT_TRUE(bb2 == bb);
    ASSERT_TRUE(pb2 == pb);
}

TEST(cold_hash_map, blob_map_erase_compaction_byte_identical) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodBlobMap m(&lb, &d);
    const int32_t N = 1200;                            // key/blob/pool span pages
    for (int32_t k = 0; k < N; ++k)
        assignBlobs(m, k, { "rec" + std::to_string(k),
                            std::string(static_cast<size_t>(k % 13), 'z') });

    const int32_t removed = m.eraseBlobIf([](int32_t k) { return k % 3 == 0; });
    int32_t expect = 0;
    for (int32_t k = 0; k < N; ++k) if (k % 3 == 0) ++expect;
    ASSERT_EQ(removed, expect);
    ASSERT_EQ(m.count(), N - expect);

    // Compacted columns are byte-identical to a from-scratch insert of survivors.
    gl::LbArena lb2(&g);
    gl::DirtyState d2 = gl::DirtyState::Clean;
    PodBlobMap fresh(&lb2, &d2);
    for (int32_t k = 0; k < N; ++k) {
        if (k % 3 == 0) continue;
        assignBlobs(fresh, k, { "rec" + std::to_string(k),
                                std::string(static_cast<size_t>(k % 13), 'z') });
    }
    std::vector<char> mk, mr, mb, mp, fk, fr, fb, fp;
    m.keyStore().appendKeyBytes(mk, 0);
    m.appendRunStartBytes(mr, 0);
    m.appendBlobStartBytes(mb, 0);
    m.appendBlobPoolBytes(mp, 0);
    fresh.keyStore().appendKeyBytes(fk, 0);
    fresh.appendRunStartBytes(fr, 0);
    fresh.appendBlobStartBytes(fb, 0);
    fresh.appendBlobPoolBytes(fp, 0);
    ASSERT_TRUE(mk == fk);
    ASSERT_TRUE(mr == fr);
    ASSERT_TRUE(mb == fb);
    ASSERT_TRUE(mp == fp);
}

TEST(cold_hash_map, blob_map_copy_from_deep_copy) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    PodBlobMap a(&lb, &d);
    for (int32_t k = 0; k < 20; ++k)
        assignBlobs(a, k * 3, { "p" + std::to_string(k), "q" + std::to_string(k) });

    gl::LbArena lb2(&g);
    gl::DirtyState d2 = gl::DirtyState::Clean;
    PodBlobMap b(&lb2, &d2);
    b.copyFrom(a);
    ASSERT_EQ(b.count(), a.count());
    ASSERT_EQ(b.blobCount(), a.blobCount());
    for (int32_t id = 1; id <= a.count(); ++id) {
        ASSERT_EQ(readBlobStr(b, id, 0), readBlobStr(a, id, 0));
        ASSERT_EQ(readBlobStr(b, id, 1), readBlobStr(a, id, 1));
    }
    // Mutating the copy leaves the source untouched (separate arenas).
    assignBlobs(b, 0, { "changed" });
    ASSERT_EQ(readBlobStr(a, a.lookup(0), 0), "p0");
}

// ---------------------------------------------------------------------------
// PagedHashIndex two-level directory spill.
//
// A 256-byte page holds 64 int32 slots/page AND dirCap = 256/4 = 64 data-page
// vids per directory page, so the single directory page caps at 64 * 64 = 4096
// slots; past that reset() builds a two-level directory (L2 root -> level-1
// directory pages -> data pages). The former single-level reset() wrote dir[p]
// past the directory page once numPages_ > 64, silently corrupting the arena;
// these exercise the fix (and its hard overflow assert) at a small page.
// ---------------------------------------------------------------------------

namespace {
    const gl::StaticMemoryConfig kHashSpillCfg{ 1 << 22, 1 << 18, 256 };
    constexpr int32_t kHashSlotsPerPage = 64;                            // 256 / 4
    constexpr int32_t kHashDirCap = 64;
    constexpr int32_t kHashSingleCap = kHashSlotsPerPage * kHashDirCap;  // 4096
}

TEST(paged_hash_index, two_level_reset_reads_all_zero) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena a(&g);
    ASSERT_EQ(a.pageBytes(), 256);
    gl::PagedHashIndex idx(&a);
    idx.reset(20000);                              // ~313 data pages -> two-level
    ASSERT_EQ(idx.capacity(), 20000);
    ASSERT_TRUE(a.livePages() > kHashDirCap + 1);  // beyond a single directory page
    for (int32_t i = 0; i < 20000; ++i) ASSERT_EQ(idx.at(i), 0);
}

TEST(paged_hash_index, two_level_in_place_set_and_at) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena a(&g);
    gl::PagedHashIndex idx(&a);
    idx.reset(20000);
    const int32_t probe[] = { 0, 1, 63, 64, 65, 4095, 4096, 4097,
                              8192, 12345, 19999 };
    for (int32_t i : probe) idx.set(i, i + 7);
    for (int32_t i : probe) ASSERT_EQ(idx.at(i), i + 7);
    ASSERT_EQ(idx.at(2), 0);                        // untouched slots stay zero
    ASSERT_EQ(idx.at(5000), 0);
    ASSERT_EQ(idx.at(19998), 0);
}

TEST(paged_hash_index, two_level_full_round_trip) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena a(&g);
    gl::PagedHashIndex idx(&a);
    idx.reset(8192);                               // 128 data pages -> two-level
    for (int32_t i = 0; i < 8192; ++i) idx.set(i, i * 7 + 1);
    for (int32_t i = 0; i < 8192; ++i) ASSERT_EQ(idx.at(i), i * 7 + 1);
}

TEST(paged_hash_index, two_level_clear_frees_all_pages) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena a(&g);
    gl::PagedHashIndex idx(&a);
    idx.reset(20000);
    ASSERT_TRUE(a.livePages() > 0);
    idx.clear();
    ASSERT_EQ(idx.capacity(), 0);
    ASSERT_TRUE(idx.empty());
    ASSERT_EQ(a.livePages(), 0);                   // every page (data + L1 + root) freed
}

TEST(paged_hash_index, two_level_reset_shrink_then_grow) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena a(&g);
    gl::PagedHashIndex idx(&a);
    idx.reset(20000);                              // two-level
    idx.set(19999, 42);
    ASSERT_EQ(idx.at(19999), 42);
    idx.reset(100);                                // wholesale rebuild -> single directory
    ASSERT_EQ(idx.capacity(), 100);
    for (int32_t i = 0; i < 100; ++i) ASSERT_EQ(idx.at(i), 0);   // fresh, all zero
    idx.reset(20000);                              // back to two-level
    ASSERT_EQ(idx.capacity(), 20000);
    for (int32_t i = 0; i < 20000; ++i) ASSERT_EQ(idx.at(i), 0);
}

TEST(paged_hash_index, two_level_promotion_boundary_page_accounting) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena a(&g);
    gl::PagedHashIndex idx(&a);
    idx.reset(kHashSingleCap);                     // 4096 slots -> 64 data pages, single dir
    ASSERT_EQ(a.livePages(), kHashDirCap + 1);     // 64 data + 1 directory page
    idx.reset(kHashSingleCap + 1);                 // 4097 -> 65 data pages -> two-level
    ASSERT_EQ(a.livePages(), kHashDirCap + 4);     // 65 data + 2 L1 + 1 root = 68
}

TEST(paged_hash_index, two_level_promotion_records_telemetry) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena a(&g);
    gl::PagedHashIndex idx(&a);
    ASSERT_EQ(g.twoLevelPromotions(), 0);          // fresh manager
    idx.reset(kHashSingleCap);                     // single directory, no promotion
    ASSERT_EQ(g.twoLevelPromotions(), 0);
    idx.reset(20000);                              // two-level -> records the promotion
    ASSERT_TRUE(g.twoLevelPromotions() >= 1);
    ASSERT_TRUE(g.peakPagesHeld() >= 313);         // ceil(20000 / 64) data pages
}

// ---------------------------------------------------------------------------
// "The vector spills inside a map": each cold-map variant populated at a small
// page until its backing PagedVector column AND the PagedHashIndex bucket array
// go two-level. The map stays correct, and g.twoLevelPromotions() > 0 confirms
// the spill fired. Before the PagedHashIndex two-level fix the bucket array
// silently corrupted the arena at these sizes — this is the regression guard.
// ---------------------------------------------------------------------------

TEST(cold_hash_map, pod_set_two_level_spill) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSet s(&lb, &dirty);
    const int32_t N = 6000;                        // keys_ (int32) past 4096 -> two-level
    for (int32_t i = 0; i < N; ++i) ASSERT_EQ(s.mint(7 * i + 3), i + 1);
    ASSERT_EQ(s.count(), N);
    ASSERT_TRUE(g.twoLevelPromotions() > 0);       // the key column / bucket index spilled
    for (int32_t i = 0; i < N; ++i) {
        ASSERT_EQ(s.lookup(7 * i + 3), i + 1);
        ASSERT_EQ(s.decode(i + 1), 7 * i + 3);
    }
    ASSERT_EQ(s.lookup(2), 0);                      // a genuine miss
    // Deload byte-identity across the spill: dump keys, reload on a fresh arena,
    // rebuild the index, re-dump -> byte-identical.
    std::vector<char> kb; s.keyStore().appendKeyBytes(kb, 0);
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodSet t(&lb2, &dirty2);
    t.keyStore().bulkLoadKeyBytes(kb.data(), s.count());
    t.rebuildIndex();
    ASSERT_EQ(t.count(), s.count());
    for (int32_t i = 0; i < N; ++i) ASSERT_EQ(t.lookup(7 * i + 3), i + 1);
    std::vector<char> kb2; t.keyStore().appendKeyBytes(kb2, 0);
    ASSERT_TRUE(kb2 == kb);
}

TEST(cold_hash_map, pod_map_two_level_spill) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMap m(&lb, &dirty);
    const int32_t N = 5000;                         // values_ (int64) past 2049 -> two-level
    for (int32_t i = 0; i < N; ++i)
        ASSERT_EQ(m.insert(11 * i + 1, 100000LL + i), i + 1);
    ASSERT_EQ(m.count(), N);
    ASSERT_TRUE(g.twoLevelPromotions() > 0);
    for (int32_t i = 0; i < N; ++i) {
        const int64_t* v = m.find(11 * i + 1);
        ASSERT_TRUE(v != nullptr);
        ASSERT_EQ(*v, 100000LL + i);
        ASSERT_EQ(m.valueAt(i + 1), 100000LL + i);
    }
    ASSERT_TRUE(m.find(7) == nullptr);
    // Deload byte-identity (key + value columns).
    std::vector<char> kb, vb;
    m.keyStore().appendKeyBytes(kb, 0);
    m.appendValueBytes(vb, 0);
    gl::LbArena lb2(&g);
    gl::DirtyState dirty2 = gl::DirtyState::Clean;
    PodMap n(&lb2, &dirty2);
    n.keyStore().bulkLoadKeyBytes(kb.data(), m.count());
    n.bulkLoadValueBytes(vb.data(), m.count());
    n.rebuildIndex();
    ASSERT_EQ(n.count(), m.count());
    for (int32_t i = 0; i < N; ++i) {
        const int64_t* v = n.find(11 * i + 1);
        ASSERT_TRUE(v != nullptr && *v == 100000LL + i);
    }
    std::vector<char> kb2, vb2;
    n.keyStore().appendKeyBytes(kb2, 0);
    n.appendValueBytes(vb2, 0);
    ASSERT_TRUE(kb2 == kb);
    ASSERT_TRUE(vb2 == vb);
}

TEST(cold_hash_map, multimap_two_level_spill) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMulti mm(&lb, &dirty);
    const int32_t K = 6000;                         // 6000 keys, one value each
    for (int32_t i = 0; i < K; ++i)
        ASSERT_EQ(mm.appendToTail(5 * i + 2, i * 3), i + 1);
    ASSERT_EQ(mm.count(), K);
    ASSERT_EQ(mm.valueCount(), K);
    ASSERT_TRUE(g.twoLevelPromotions() > 0);
    for (int32_t i = 0; i < K; ++i) {
        ASSERT_EQ(mm.runLen(i + 1), 1);
        ASSERT_EQ(mm.valueAt(i + 1, 0), i * 3);
        ASSERT_EQ(mm.lookup(5 * i + 2), i + 1);
    }
    ASSERT_FALSE(mm.contains(3));
}

TEST(cold_hash_map, set_map_two_level_spill) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSetMap m(&lb, &dirty);
    const int32_t K = 6000;
    for (int32_t i = 0; i < K; ++i)
        ASSERT_EQ(m.insertSorted(9 * i + 4, i + 1000), i + 1);
    ASSERT_EQ(m.count(), K);
    ASSERT_EQ(m.valueCount(), K);
    ASSERT_TRUE(g.twoLevelPromotions() > 0);
    for (int32_t i = 0; i < K; ++i) {
        ASSERT_EQ(m.runLen(i + 1), 1);
        ASSERT_EQ(m.valueAt(i + 1, 0), i + 1000);
        ASSERT_TRUE(m.setContains(9 * i + 4, i + 1000));
        ASSERT_FALSE(m.setContains(9 * i + 4, 999999));
    }
}

// Interior splices of long runs across many pages: sorted inserts and
// grow/shrink whole-set replaces move every later run through the bulk value
// shift and the bulk run-start adjust; runs and totals match a std oracle.
TEST(cold_hash_map, set_map_multipage_splices_match_oracle) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodSetMap m(&lb, &dirty);
    std::map<int32_t, std::set<int32_t>> oracle;
    std::vector<int32_t> keyOrder;
    uint32_t rng = 0x2468ace1u;
    auto nx = [&]() { rng = rng * 1664525u + 1013904223u; return rng; };
    for (int32_t step = 0; step < 4000; ++step) {
        const int32_t key = static_cast<int32_t>(nx() % 60u);
        if (oracle.find(key) == oracle.end()) keyOrder.push_back(key);
        if (nx() % 8u != 0u) {
            const int32_t value = static_cast<int32_t>(nx() % 5000u);
            m.insertSorted(key, value);
            oracle[key].insert(value);
        } else {
            std::set<int32_t> next;
            const uint32_t n = nx() % 400u;
            for (uint32_t j = 0; j < n; ++j) next.insert(static_cast<int32_t>(nx() % 5000u));
            const std::vector<int32_t> run(next.begin(), next.end());
            m.assignSet(key, run.data(), static_cast<int32_t>(run.size()));
            oracle[key] = next;
        }
    }
    ASSERT_TRUE(g.twoLevelPromotions() > 0);
    int32_t total = 0;
    ASSERT_EQ(m.count(), static_cast<int32_t>(keyOrder.size()));
    for (int32_t i = 0; i < static_cast<int32_t>(keyOrder.size()); ++i) {
        const std::set<int32_t>& want = oracle[keyOrder[i]];
        ASSERT_EQ(m.lookup(keyOrder[i]), i + 1);
        ASSERT_EQ(m.runLen(i + 1), static_cast<int32_t>(want.size()));
        int32_t j = 0;
        for (const int32_t v : want) ASSERT_EQ(m.valueAt(i + 1, j++), v);
        total += static_cast<int32_t>(want.size());
    }
    ASSERT_EQ(m.valueCount(), total);
}

TEST(cold_hash_map, blob_map_two_level_blobpool_spill) {
    gl::GlobalMemoryManager g;
    g.init(kHashSpillCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodBlobMap m(&lb, &dirty);
    const int32_t K = 80;                           // 80 * 1024-byte blob -> char pool two-level
    for (int32_t i = 0; i < K; ++i) {
        std::string s(1024, 'z');
        s[0] = static_cast<char>('a' + (i % 26));   // a distinguishable first byte
        ASSERT_EQ(assignBlobs(m, 100 + i, { s }), i + 1);
    }
    ASSERT_EQ(m.count(), K);
    ASSERT_TRUE(g.twoLevelPromotions() > 0);         // blobPool (char) spilled
    for (int32_t i = 0; i < K; ++i) {
        ASSERT_EQ(m.runLen(i + 1), 1);
        const std::string got = readBlobStr(m, i + 1, 0);
        ASSERT_EQ(static_cast<int32_t>(got.size()), 1024);
        ASSERT_EQ(got[0], static_cast<char>('a' + (i % 26)));
        ASSERT_EQ(got[1023], 'z');
        ASSERT_EQ(m.lookup(100 + i), i + 1);
    }
}

// HashMap::insertEpoch: a fresh mint bumps it, a lookup or a re-mint of an
// existing key does not, resetToFresh does.
TEST(cold_hash_map, insert_epoch_tracks_key_set_growth_only) {
    gl::ScratchArena& a = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::PodKeyStore<int32_t>> s(&a, &d);
    s.resetToFresh();
    const uint32_t e0 = s.insertEpoch();
    ASSERT_EQ(s.mint(7), 1);
    ASSERT_EQ(s.insertEpoch(), e0 + 1);
    ASSERT_EQ(s.lookup(7), 1);
    ASSERT_EQ(s.insertEpoch(), e0 + 1);       // lookup: no bump
    ASSERT_EQ(s.mint(7), 1);
    ASSERT_EQ(s.insertEpoch(), e0 + 1);       // existing key: no bump
    ASSERT_EQ(s.mint(9), 2);
    ASSERT_EQ(s.insertEpoch(), e0 + 2);
    s.resetToFresh();
    ASSERT_EQ(s.insertEpoch(), e0 + 3);       // key set replaced: bump
    ASSERT_EQ(s.count(), 0);
}

// A find-then-insert that probes once (lookupForInsert + insertAt / mintAt) gives
// the same ids, values, index size and epochs as lookup + insert / mint, across
// index growth and an erase-driven rebuild, for POD and byte keys.
TEST(cold_hash_map, insert_at_probed_slot_matches_insert) {
    gl::GlobalMemoryManager g;
    g.init(kMapTestCfg);
    gl::LbArena lb(&g);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMap reference(&lb, &dirty);
    PodMap probed(&lb, &dirty);
    uint32_t state = 12345u;
    const auto nextKey = [&state]() {
        state = state * 1664525u + 1013904223u;
        return static_cast<int32_t>((state >> 8) % 2500u);
    };
    for (int32_t step = 0; step < 6000; ++step) {
        const int32_t key = nextKey();
        const int64_t value = 1000LL * step + key;
        const int32_t referenceId = reference.lookup(key);
        if (referenceId == 0) reference.insert(key, value);
        else reference.setValueAt(referenceId, value);
        PodMap::InsertHint hint;
        const int32_t probedId = probed.lookupForInsert(key, hint);
        ASSERT_EQ(probedId, referenceId);
        if (probedId == 0) ASSERT_EQ(probed.insertAt(key, value, hint), reference.count());
        else probed.setValueAt(probedId, value);
        if (step == 3000) {
            const auto odd = [](int32_t k) { return (k % 3) == 1; };
            ASSERT_EQ(reference.eraseIf(odd), probed.eraseIf(odd));
        }
    }
    ASSERT_EQ(probed.count(), reference.count());
    ASSERT_EQ(probed.insertEpoch(), reference.insertEpoch());
    ASSERT_EQ(probed.indexBytes(), reference.indexBytes());
    for (int32_t key = 0; key < 2500; ++key) {
        const int32_t id = reference.lookup(key);
        ASSERT_EQ(probed.lookup(key), id);
        if (id != 0) ASSERT_EQ(probed.valueAt(id), reference.valueAt(id));
    }

    gl::ColdHashSet<gl::BytesKeyStore> names(&lb, &dirty);
    gl::ColdHashSet<gl::BytesKeyStore> namesProbed(&lb, &dirty);
    for (int32_t step = 0; step < 900; ++step) {
        const std::string text = "(p[" + std::to_string(nextKey() % 400) + "])";
        const gl::StrSpan key(text);
        const int32_t id = names.mint(key);
        gl::ColdHashSet<gl::BytesKeyStore>::InsertHint hint;
        int32_t probedId = namesProbed.lookupForInsert(key, hint);
        if (probedId == 0) probedId = namesProbed.mintAt(key, hint);
        ASSERT_EQ(probedId, id);
    }
    ASSERT_EQ(namesProbed.count(), names.count());
    ASSERT_EQ(namesProbed.insertEpoch(), names.insertEpoch());
    ASSERT_EQ(namesProbed.indexBytes(), names.indexBytes());
}


TEST(cold_hash_map, shared_native_keys_keep_local_identity_and_canonical_bytes) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), scratch(&global), otherArena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::SharedBytePool pool(&arena, &dirty);
    using Bytes = gl::ColdHashSet<gl::BytesKeyStore>;
    Bytes first(&arena, &dirty), second(&arena, &dirty), oracle(&otherArena, &dirty);
    first.keyStore().bindSharedBytes(pool);
    second.keyStore().bindSharedBytes(pool);
    ASSERT_TRUE(first.keyStore().sharesBytes());
    ASSERT_FALSE(oracle.keyStore().sharesBytes());
    const std::string dead(1700, 'd'), alpha(1700, 'a'), beta(600, 'b');
    first.mint(gl::StrSpan(dead));
    const int firstAlpha = first.mint(gl::StrSpan(alpha));
    const int secondAlpha = second.mint(gl::StrSpan(alpha));
    second.mint(gl::StrSpan(beta));
    oracle.mint(gl::StrSpan(alpha)); oracle.mint(gl::StrSpan(beta));
    ASSERT_EQ(firstAlpha, 2); ASSERT_EQ(secondAlpha, 1);
    ASSERT_EQ(first.keyAt(firstAlpha).ptr, second.keyAt(secondAlpha).ptr);
    ASSERT_EQ(first.lookup(gl::StrSpan(beta)), 0);
    ASSERT_EQ(pool.count(), 3);
    ASSERT_EQ(pool.logicalByteCount(), static_cast<int64_t>(dead.size() + alpha.size() + beta.size()));
    ASSERT_TRUE(pool.liveBytes() >= pool.logicalByteCount());
    ASSERT_TRUE(pool.indexBytes() > 0);
    std::vector<char> lengths, bytes, expectedLengths, expectedBytes;
    second.appendLengthBytes(lengths, 0); second.appendContentBytes(bytes, 0);
    oracle.appendLengthBytes(expectedLengths, 0); oracle.appendContentBytes(expectedBytes, 0);
    ASSERT_TRUE(lengths == expectedLengths && bytes == expectedBytes);
    ASSERT_EQ(second.decodeString(1), alpha);
    first.resetToFresh();
    pool.collect(scratch);
    ASSERT_EQ(pool.count(), 2);
    ASSERT_EQ(second.lookup(gl::StrSpan(alpha)), 1);
    ASSERT_EQ(second.lookup(gl::StrSpan(beta)), 2);
    ASSERT_EQ(second.decodeString(1), alpha);
    ASSERT_EQ(second.decodeString(2), beta);
    ASSERT_EQ(pool.logicalByteCount(), static_cast<int64_t>(alpha.size() + beta.size()));
    pool.collect(scratch);
    Bytes reloaded(&arena, &dirty), copied(&arena, &dirty), plain(&otherArena, &dirty);
    reloaded.keyStore().bindSharedBytes(pool); copied.keyStore().bindSharedBytes(pool);
    reloaded.stageLengths(lengths.data(), second.count());
    reloaded.consumeStagedLengths(bytes.data(), bytes.size());
    copied.copyFrom(oracle); plain.copyFrom(second);
    ASSERT_EQ(reloaded.keyAt(1).ptr, second.keyAt(1).ptr);
    ASSERT_EQ(copied.keyAt(1).ptr, second.keyAt(1).ptr);
    ASSERT_EQ(plain.decodeString(2), beta);
    std::vector<char> tail;
    reloaded.appendContentBytesFromByte(tail, alpha.size());
    ASSERT_TRUE(tail == std::vector<char>(beta.begin(), beta.end()));
    ASSERT_EQ(second.eraseIf([&](gl::StrSpan key) { return key.len == static_cast<int>(alpha.size()); }), 1);
    ASSERT_EQ(second.lookup(gl::StrSpan(beta)), 1);
    pool.collect(scratch);
    ASSERT_EQ(reloaded.lookup(gl::StrSpan(alpha)), 1);
    first.release(); second.release(); reloaded.release(); copied.release();
    pool.collect(scratch);
    ASSERT_EQ(pool.count(), 0); ASSERT_EQ(pool.logicalByteCount(), 0);
    pool.release();
}

TEST(cold_hash_map, shared_normalized_values_match_owning_run_operations) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), scratch(&global), otherArena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::SharedBytePool pool(&arena, &dirty);
    gl::ColdHashSet<gl::BytesKeyStore> keys(&arena, &dirty);
    keys.keyStore().bindSharedBytes(pool);
    PodBlobMap shared(&arena, &dirty), oracle(&otherArena, &dirty);
    shared.bindSharedBlobBytes(pool);
    const std::string a(1700, 'a'), b(800, 'b'), c(900, 'c');
    keys.mint(gl::StrSpan(a));
    const auto same = [&]() {
        std::vector<char> x, y;
        shared.appendRunStartBytes(x, 0); oracle.appendRunStartBytes(y, 0); ASSERT_TRUE(x == y);
        x.clear(); y.clear();
        shared.appendBlobStartBytes(x, 0); oracle.appendBlobStartBytes(y, 0); ASSERT_TRUE(x == y);
        x.clear(); y.clear();
        shared.appendBlobPoolBytes(x, 0); oracle.appendBlobPoolBytes(y, 0); ASSERT_TRUE(x == y);
        ASSERT_EQ(shared.poolByteCount(), oracle.poolByteCount());
        ASSERT_EQ(shared.residentPoolByteCount(), shared.blobCount() * 4);
    };
    assignBlobs(shared, 1, {a, b}); assignBlobs(oracle, 1, {a, b});
    assignBlobs(shared, 2, {a, c}); assignBlobs(oracle, 2, {a, c}); same();
    ASSERT_EQ(pool.count(), 3);
    std::vector<char> buffer;
    int32_t length = 0;
    const char* bytes = shared.peekBlobContiguous(1, 0, length, buffer);
    ASSERT_EQ(bytes, keys.keyAt(1).ptr);
    assignBlobs(shared, 1, {c}); assignBlobs(oracle, 1, {c}); same();
    assignBlobs(shared, 1, {b, a, c}); assignBlobs(oracle, 1, {b, a, c}); same();
    std::vector<char> values;
    shared.blobAt(1, 1, values); ASSERT_TRUE(values == std::vector<char>(a.begin(), a.end()));
    PodBlobMap copied(&arena, &dirty), plain(&otherArena, &dirty), loaded(&arena, &dirty);
    copied.bindSharedBlobBytes(pool); loaded.bindSharedBlobBytes(pool);
    copied.copyFrom(oracle); plain.copyFrom(shared);
    std::vector<char> wireKeys, wireRuns, wireStarts, wirePool, copiedPool;
    shared.keyStore().appendKeyBytes(wireKeys, 0); shared.appendRunStartBytes(wireRuns, 0);
    shared.appendBlobStartBytes(wireStarts, 0); shared.appendBlobPoolBytes(wirePool, 0);
    loaded.keyStore().bulkLoadKeyBytes(wireKeys.data(), shared.count()); loaded.rebuildIndex();
    loaded.bulkLoadRunStartBytes(wireRuns.data(), shared.count());
    loaded.bulkLoadBlobStartBytes(wireStarts.data(), shared.blobCount());
    loaded.bulkLoadBlobPoolBytes(wirePool.data(), wirePool.size());
    loaded.appendBlobPoolBytes(copiedPool, 0); ASSERT_TRUE(copiedPool == wirePool);
    copiedPool.clear(); plain.appendBlobPoolBytes(copiedPool, 0); ASSERT_TRUE(copiedPool == wirePool);
    ASSERT_EQ(shared.eraseBlobIf([](int key) { return key == 1; }), 1);
    ASSERT_EQ(oracle.eraseBlobIf([](int key) { return key == 1; }), 1); same();
    pool.collect(scratch); same();
    assignBlobs(shared, 3, {}); assignBlobs(oracle, 3, {}); same();
    shared.assignAllRunsGenerated(shared.count(), [&](int32_t id, auto& sink) {
        if (id == 1) sink.unchanged(id); else sink.blob(b.data(), static_cast<int32_t>(b.size()));
    }, scratch);
    oracle.assignAllRunsGenerated(oracle.count(), [&](int32_t id, auto& sink) {
        if (id == 1) sink.unchanged(id); else sink.blob(b.data(), static_cast<int32_t>(b.size()));
    }, scratch); same();
    shared.release(); copied.release(); loaded.release(); keys.release(); pool.release();
}

TEST(cold_hash_map, shared_native_key_assertion_probe) {
    std::string mode;
#ifdef _WIN32
    char* value = nullptr; size_t length = 0;
    ASSERT_EQ(_dupenv_s(&value, &length, "GL_TEST_INVALID_SHARED_NATIVE_KEYS"), 0);
    if (value != nullptr) { mode = value; std::free(value); }
#else
    if (const char* value = std::getenv("GL_TEST_INVALID_SHARED_NATIVE_KEYS")) mode = value;
#endif
    gl::GlobalMemoryManager global; global.init(kMapTestCfg);
    gl::LbArena arena(&global), otherArena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::SharedBytePool pool(&arena, &dirty);
    gl::ColdHashSet<gl::BytesKeyStore> keys(&arena, &dirty), foreign(&otherArena, &dirty);
    if (mode == "nonempty") {
        keys.mint(gl::StrSpan("key", 3)); keys.keyStore().bindSharedBytes(pool);
    } else if (mode == "cross_block") {
        foreign.keyStore().bindSharedBytes(pool);
    } else if (mode == "live_release") {
        keys.keyStore().bindSharedBytes(pool); keys.mint(gl::StrSpan("key", 3)); pool.release();
    } else if (mode == "oversized_normalized") {
        PodBlobMap values(&arena, &dirty); values.bindSharedBlobBytes(pool);
        assignBlobs(values, 1, {std::string(arena.pageBytes() + 1, 'x')});
    } else if (mode == "empty_value") {
        PodBlobMap values(&arena, &dirty); values.bindSharedBlobBytes(pool); assignBlobs(values, 1, {""});
    }
}


TEST(cold_hash_map, shared_native_last_client_reset_releases_and_can_refill) {
    gl::GlobalMemoryManager global; global.init(kMapTestCfg);
    gl::LbArena arena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::SharedBytePool pool(&arena, &dirty);
    gl::ColdHashSet<gl::BytesKeyStore> keys(&arena, &dirty);
    PodBlobMap values(&arena, &dirty);
    keys.keyStore().bindSharedBytes(pool); values.bindSharedBlobBytes(pool);
    pool.collect(scratch); pool.releaseIfUnused();
    keys.mint(gl::StrSpan("payload", 7)); assignBlobs(values, 1, {"payload"});
    keys.resetToFresh(); pool.releaseIfUnused();
    ASSERT_EQ(pool.count(), 1); ASSERT_TRUE(pool.liveBytes() > 0);
    values.resetToFresh();
    ASSERT_EQ(pool.count(), 0); ASSERT_EQ(pool.liveBytes(), 0);
    keys.mint(gl::StrSpan("second", 6)); assignBlobs(values, 1, {"second"});
    values.eraseBlobIf([](int) { return true; }); ASSERT_EQ(pool.count(), 1);
    keys.eraseIf([](gl::StrSpan) { return true; });
    ASSERT_EQ(pool.count(), 0); ASSERT_EQ(pool.liveBytes(), 0);
    keys.mint(gl::StrSpan()); ASSERT_EQ(pool.count(), 1);
    ASSERT_EQ(keys.decodeString(1), std::string());
    keys.release(); ASSERT_EQ(pool.count(), 0);
}


TEST(cold_hash_map, shared_rule_values_preserve_large_records_and_private_lifetimes) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{16 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), otherArena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::SharedBytePool pool(&arena, &dirty);
    PodBlobMap first(&arena, &dirty), peer(&arena, &dirty), oracle(&otherArena, &dirty);
    first.bindSharedBlobBytes(pool, true); peer.bindSharedBlobBytes(pool, true);
    const std::string small(1500, 's'), large(12000, 'l'), replacement(14000, 'r');
    assignBlobs(first, 1, {small, large}); assignBlobs(oracle, 1, {small, large});
    assignBlobs(peer, 7, {small});
    ASSERT_EQ(pool.count(), 1);
    ASSERT_EQ(first.residentPoolByteCount(), 12008);
    const auto identical = [&]() {
        std::vector<char> a, b;
        first.appendRunStartBytes(a, 0); oracle.appendRunStartBytes(b, 0); ASSERT_TRUE(a == b);
        a.clear(); b.clear(); first.appendBlobStartBytes(a, 0); oracle.appendBlobStartBytes(b, 0); ASSERT_TRUE(a == b);
        a.clear(); b.clear(); first.appendBlobPoolBytes(a, 0); oracle.appendBlobPoolBytes(b, 0); ASSERT_TRUE(a == b);
    };
    identical();
    std::vector<char> a, b;
    int32_t n = 0, m = 0;
    ASSERT_EQ(first.peekBlobContiguous(1, 0, n, a), peer.peekBlobContiguous(1, 0, m, b));
    first.blobAt(1, 1, a); ASSERT_TRUE(a == std::vector<char>(large.begin(), large.end()));
    assignBlobs(first, 1, {replacement, small}); assignBlobs(oracle, 1, {replacement, small});
    pool.collect(scratch); identical();
    ASSERT_EQ(first.residentPoolByteCount(), 14008);
    PodBlobMap copied(&arena, &dirty), plain(&otherArena, &dirty), loaded(&arena, &dirty);
    copied.bindSharedBlobBytes(pool, true); loaded.bindSharedBlobBytes(pool, true);
    copied.copyFrom(first); plain.copyFrom(first);
    std::vector<char> keys, runs, starts, bytes;
    first.keyStore().appendKeyBytes(keys, 0); first.appendRunStartBytes(runs, 0);
    first.appendBlobStartBytes(starts, 0); first.appendBlobPoolBytes(bytes, 0);
    loaded.keyStore().bulkLoadKeyBytes(keys.data(), first.count()); loaded.rebuildIndex();
    loaded.bulkLoadRunStartBytes(runs.data(), first.count());
    loaded.bulkLoadBlobStartBytes(starts.data(), first.blobCount());
    loaded.bulkLoadBlobPoolBytes(bytes.data(), bytes.size());
    for (auto* map : {&copied, &plain, &loaded}) {
        a.clear(); map->appendBlobPoolBytes(a, 0); ASSERT_TRUE(a == bytes);
    }
    const auto grownRun = [&](int32_t, auto& sink) {
        sink.blob(replacement.data(), replacement.size());
        sink.blob(small.data(), small.size());
        sink.blob(large.data(), large.size());
    };
    first.assignAllRunsGenerated(first.count(), grownRun, scratch);
    oracle.assignAllRunsGenerated(oracle.count(), grownRun, scratch);
    pool.collect(scratch); identical(); ASSERT_EQ(first.residentPoolByteCount(), 26012);
    first.assignAllRunsViaScratch(first.count(), [&](int32_t, auto& sink) { sink.blob(small.data(), small.size()); }, scratch);
    oracle.assignAllRunsViaScratch(oracle.count(), [&](int32_t, auto& sink) { sink.blob(small.data(), small.size()); }, scratch);
    pool.collect(scratch); identical(); ASSERT_EQ(first.residentPoolByteCount(), 4);
    first.release(); copied.release(); loaded.release();
    peer.blobAt(1, 0, a); ASSERT_TRUE(a == std::vector<char>(small.begin(), small.end()));
    peer.release(); pool.release(); ASSERT_EQ(pool.count(), 0);
}


TEST(cold_hash_map, three_quarter_growth_preserves_collision_chains_and_ids) {
    gl::GlobalMemoryManager global;
    global.init(kMapTestCfg);
    gl::LbArena arena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    PodMap ordinary(&arena, &dirty), hinted(&arena, &dirty);
    std::vector<int32_t> keys;
    for (int32_t key = 0; keys.size() < 192; ++key) {
        const auto hash = gl::hashSpan(gl::StrSpan(reinterpret_cast<const char*>(&key), sizeof(key)));
        if ((hash & 63) == 63) keys.push_back(key);
    }
    for (int32_t i = 0; i < 191; ++i) {
        PodMap::InsertHint hint;
        ASSERT_EQ(hinted.lookupForInsert(keys[i], hint), 0);
        ASSERT_EQ(hinted.insertAt(keys[i], int64_t(i + 100), hint), i + 1);
        ASSERT_EQ(ordinary.insert(keys[i], int64_t(i + 100)), i + 1);
        PodMap::InsertHint capacity;
        ASSERT_EQ(ordinary.lookupForInsert(keys[191], capacity), 0);
        const int32_t count = i + 1;
        ASSERT_EQ(capacity.capacity, count < 48 ? 64 : count < 96 ? 128 : 256);
        if (count == 47 || count == 48 || count == 95 || count == 96 || count == 191) {
            ordinary.rebuildIndex(); hinted.rebuildIndex();
            for (int32_t j = 0; j <= i; ++j) {
                ASSERT_EQ(ordinary.lookup(keys[j]), j + 1);
                ASSERT_EQ(hinted.lookup(keys[j]), j + 1);
                ASSERT_EQ(hinted.valueAt(j + 1), int64_t(j + 100));
            }
            ASSERT_EQ(hinted.lookup(keys[191]), 0);
        }
    }
    const auto remove = [&](int32_t key) { return key >= keys[24]; };
    ASSERT_EQ(ordinary.eraseIf(remove), 167);
    ASSERT_EQ(hinted.eraseIf(remove), 167);
    PodMap::InsertHint hint;
    ASSERT_EQ(ordinary.lookupForInsert(keys[191], hint), 0);
    ASSERT_EQ(hint.capacity, 64);
    for (int32_t i = 0; i < 24; ++i) {
        ASSERT_EQ(ordinary.lookup(keys[i]), i + 1);
        ASSERT_EQ(hinted.valueAt(i + 1), int64_t(i + 100));
    }
    PodMap copied(&arena, &dirty);
    copied.copyFrom(ordinary);
    ASSERT_EQ(copied.lookup(keys[191]), 0);
    ASSERT_EQ(copied.lookupForInsert(keys[0], hint), 1);
    ASSERT_EQ(hint.capacity, 64);
}

TEST(cold_hash_map, prefix_keys_keep_ids_hashes_canonical_bytes_and_caller_lifetimes) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{32 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), scratch(&global), copies(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PrefixKeyMetadata metadata(&arena, &dirty);
    gl::ColdStringTable packed(&arena, &dirty), oracle(&arena, &dirty);
    packed.keyStore().bindPrefixes(metadata);
    std::vector<std::string> expected;
    expected.push_back("");
    for (int i = 0; i < 2300; ++i)
        expected.push_back(std::string(600, 'x') + std::to_string((i * 137) % 2300) + std::string(i % 11, 'z'));
    for (int i = 0; i < 1100; ++i) {
        ASSERT_EQ(packed.intern(gl::StrSpan(expected[i])), i + 1);
        oracle.intern(gl::StrSpan(expected[i]));
    }
    const auto before = packed.liveBytes();
    packed.keyStore().packPrefixes(scratch);
    ASSERT_EQ(metadata.rows.size(), 1100);
    ASSERT_TRUE(packed.liveBytes() < before);
    const gl::StrSpan firstCopy = packed.keyStore().copyView(2, copies);
    const gl::StrSpan secondCopy = packed.keyStore().copyView(3, copies);
    for (int i = 1100; i < static_cast<int>(expected.size()); ++i) {
        packed.intern(gl::StrSpan(expected[i])); oracle.intern(gl::StrSpan(expected[i]));
    }
    packed.keyStore().packPrefixes(scratch);
    ASSERT_EQ(metadata.rows.size(), static_cast<int32_t>(expected.size()));
    ASSERT_TRUE(gl::equalSpans(firstCopy, gl::StrSpan(expected[1])));
    ASSERT_TRUE(gl::equalSpans(secondCopy, gl::StrSpan(expected[2])));
    char buffer[2048];
    for (int id = 1; id <= packed.count(); ++id) {
        ASSERT_EQ(packed.lookup(gl::StrSpan(expected[id - 1])), id);
        ASSERT_EQ(packed.keyStore().copyKeyTo(id, buffer, sizeof(buffer)), static_cast<int32_t>(expected[id - 1].size()));
        ASSERT_TRUE(gl::equalSpans(gl::StrSpan(buffer, static_cast<int32_t>(expected[id - 1].size())), gl::StrSpan(expected[id - 1])));
        ASSERT_EQ(packed.decodeString(id), expected[id - 1]);
        const int other = id == packed.count() ? 1 : id + 1;
        const int comparison = packed.keyStore().compareKeys(id, other);
        const int expectedComparison = expected[id - 1].compare(expected[other - 1]);
        ASSERT_EQ(comparison < 0, expectedComparison < 0);
        ASSERT_EQ(comparison == 0, expectedComparison == 0);
        ASSERT_EQ(packed.keyStore().compareKeys(id, id), 0);
    }
    ASSERT_EQ(packed.lookup(gl::StrSpan("absent", 6)), 0);
    std::vector<char> lengths, bytes, expectedLengths, expectedBytes;
    packed.appendLengthBytes(lengths, 0); packed.appendContentBytes(bytes, 0);
    oracle.appendLengthBytes(expectedLengths, 0); oracle.appendContentBytes(expectedBytes, 0);
    ASSERT_TRUE(lengths == expectedLengths && bytes == expectedBytes);
    gl::ColdStringTable clone(&arena, &dirty), loaded(&arena, &dirty);
    clone.copyFrom(packed);
    loaded.keyStore().bulkLoadKeys(reinterpret_cast<const int32_t*>(lengths.data()), bytes.data(), bytes.size(), packed.count());
    loaded.rebuildIndex();
    for (int id = 1; id <= packed.count(); ++id) {
        ASSERT_EQ(clone.decodeString(id), expected[id - 1]);
        ASSERT_EQ(loaded.lookup(gl::StrSpan(expected[id - 1])), id);
    }
    packed.resetToFresh(); ASSERT_TRUE(metadata.rows.empty()); ASSERT_EQ(metadata.examinedCount, 0);
    ASSERT_EQ(packed.liveBytes(), int64_t(0));
    packed.intern(gl::StrSpan("fresh", 5)); ASSERT_EQ(packed.decodeString(1), std::string("fresh"));
    packed.release(); ASSERT_TRUE(metadata.rows.empty());
}

TEST(cold_hash_map, prefix_format_keeps_short_keys_uncompressed) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PrefixKeyMetadata metadata(&arena, &dirty);
    gl::ColdStringTable keys(&arena, &dirty);
    keys.keyStore().bindPrefixes(metadata);
    keys.keyStore().packPrefixes(scratch);
    ASSERT_EQ(metadata.examinedCount, 0);
    for (int i = 0; i < 1100; ++i) {
        const std::string text = std::to_string(i);
        keys.intern(gl::StrSpan(text));
    }
    keys.keyStore().packPrefixes(scratch);
    ASSERT_TRUE(metadata.rows.empty());
    ASSERT_EQ(keys.view(1).toStdString(), std::string("0"));
}


TEST(cold_hash_map, prefix_key_invalid_view_assertion_probe) {
    char* mode = nullptr; size_t length = 0;
#ifdef _WIN32
    ASSERT_EQ(_dupenv_s(&mode, &length, "GL_TEST_INVALID_PREFIX_KEY"), 0);
#else
    mode = std::getenv("GL_TEST_INVALID_PREFIX_KEY");
#endif
    if (mode == nullptr) return;
    const bool shortBuffer = std::strcmp(mode, "capacity") == 0;
#ifdef _WIN32
    std::free(mode);
#endif
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PrefixKeyMetadata metadata(&arena, &dirty);
    gl::ColdStringTable keys(&arena, &dirty);
    keys.keyStore().bindPrefixes(metadata);
    for (int i = 0; i < 1100; ++i) keys.intern(gl::StrSpan(std::string(300, 'x') + std::to_string(i)));
    keys.keyStore().packPrefixes(scratch);
    if (shortBuffer) { char buffer[1]; keys.keyStore().copyKeyTo(1, buffer, 1); }
    else keys.view(1);
    ASSERT_TRUE(false);
}

TEST(cold_hash_map, cold_string_location_packs_length_and_hash_tag) {
    static_assert(sizeof(gl::ColdStringLocation) == 8);
    ASSERT_TRUE(gl::kMaxColdStringKeyBytes >= gl::StaticMemoryConfig{}.pageBytes);
    const uint64_t hash = 0xfedcba9876543210ull;
    const gl::ColdStringLocation location =
        gl::makeColdStringLocation(123456, gl::kMaxColdStringKeyBytes, hash);
    ASSERT_EQ(location.byteStart, 123456);
    ASSERT_EQ(static_cast<int32_t>(location.len), gl::kMaxColdStringKeyBytes);
    ASSERT_EQ(location.tag, static_cast<uint32_t>(hash >> 46));
    ASSERT_EQ(gl::coldKeyTag(hash), static_cast<uint32_t>(location.tag));
    const gl::ColdStringLocation empty = gl::makeColdStringLocation(7, 0, 0);
    ASSERT_EQ(empty.byteStart, 7);
    ASSERT_EQ(static_cast<int32_t>(empty.len), 0);
    ASSERT_EQ(static_cast<uint32_t>(empty.tag), 0u);
}

TEST(cold_hash_map, byte_key_store_recomputes_stored_hashes_for_every_layout) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{32 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    std::vector<std::string> keys;
    keys.push_back("");
    for (int i = 0; i < 300; ++i)
        keys.push_back(std::string(40, 'p') + std::to_string((i * 37) % 300) + std::string(i % 7, 'q'));
    const auto check = [&](gl::ColdStringTable& table, int32_t count) {
        for (int32_t id = 1; id <= count; ++id) {
            const gl::StrSpan key(keys[id - 1]);
            ASSERT_EQ(table.keyStore().hashStored(id), gl::coldKeyHash(key));
            ASSERT_EQ(table.lookup(key), id);
        }
    };
    // Plain store: minting 300 keys crosses several index rebuilds.
    gl::ColdStringTable plain(&arena, &dirty);
    for (int32_t i = 0; i < static_cast<int32_t>(keys.size()); ++i)
        ASSERT_EQ(plain.intern(gl::StrSpan(keys[i])), i + 1);
    check(plain, static_cast<int32_t>(keys.size()));
    plain.rebuildFindIndex();
    check(plain, static_cast<int32_t>(keys.size()));
    // Prefix-packed store: hashes come from decoded predecessor chains.
    gl::PrefixKeyMetadata metadata(&arena, &dirty);
    gl::ColdStringTable packed(&arena, &dirty);
    packed.keyStore().bindPrefixes(metadata);
    for (int32_t i = 0; i < 200; ++i) packed.intern(gl::StrSpan(keys[i]));
    packed.keyStore().packPrefixes(scratch);
    for (int32_t i = 200; i < static_cast<int32_t>(keys.size()); ++i)
        packed.intern(gl::StrSpan(keys[i]));
    packed.rebuildFindIndex();
    check(packed, static_cast<int32_t>(keys.size()));
    // Shared store: hashes come from the shared payload.
    gl::SharedBytePool pool(&arena, &dirty);
    gl::ColdHashSet<gl::BytesKeyStore> shared(&arena, &dirty);
    shared.keyStore().bindSharedBytes(pool);
    for (int32_t i = 1; i < static_cast<int32_t>(keys.size()); ++i)
        ASSERT_EQ(shared.mint(gl::StrSpan(keys[i])), i);
    shared.rebuildFindIndex();
    for (int32_t id = 1; id < static_cast<int32_t>(keys.size()); ++id) {
        const gl::StrSpan key(keys[id]);
        ASSERT_EQ(shared.keyStore().hashStored(id), gl::coldKeyHash(key));
        ASSERT_EQ(shared.lookup(key), id);
    }
}

TEST(cold_hash_map, byte_key_store_tag_collision_falls_back_to_byte_comparison) {
    // Two equal-length keys whose digests share the 18-bit tag.
    std::map<uint32_t, std::string> byTag;
    std::string first, second;
    for (int i = 0; i < (1 << 22) && first.empty(); ++i) {
        std::string key = "key" + std::to_string(10000000 + i);
        const uint32_t tag = gl::coldKeyTag(gl::coldKeyHash(gl::StrSpan(key)));
        const auto found = byTag.find(tag);
        if (found != byTag.end()) { first = found->second; second = key; }
        else byTag.emplace(tag, key);
    }
    ASSERT_FALSE(first.empty());
    ASSERT_EQ(first.size(), second.size());
    ASSERT_TRUE(first != second);
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable table(&arena, &dirty);
    const int32_t firstId = table.intern(gl::StrSpan(first));
    ASSERT_TRUE(table.keyStore().equalStored(firstId, gl::StrSpan(first),
        gl::coldKeyHash(gl::StrSpan(first))));
    ASSERT_FALSE(table.keyStore().equalStored(firstId, gl::StrSpan(second),
        gl::coldKeyHash(gl::StrSpan(second))));
    ASSERT_EQ(table.lookup(gl::StrSpan(second)), 0);
    const int32_t secondId = table.intern(gl::StrSpan(second));
    ASSERT_EQ(secondId, firstId + 1);
    ASSERT_EQ(table.lookup(gl::StrSpan(first)), firstId);
    ASSERT_EQ(table.lookup(gl::StrSpan(second)), secondId);
}


namespace {
    // Canonical lengths + bytes of a byte-key set: what its deload image streams.
    std::vector<char> canonicalBytes(const gl::ColdHashSet<gl::BytesKeyStore>& set) {
        std::vector<char> image;
        set.keyStore().appendLengthBytes(image, 0);
        set.keyStore().appendContentBytes(image, 0);
        return image;
    }

    // One nomination with the digest the production scan passes along.
    void nominateBytes(gl::BatchByteCandidateRegistry& registry, gl::StrSpan bytes) {
        registry.nominate(bytes, gl::coldKeyHash(bytes));
    }

    // Nominations of one payload summed over the stripes, and the number of
    // stripes holding it (a payload lives in exactly one stripe).
    std::pair<int32_t, int32_t> nominationsOf(gl::BatchByteCandidateRegistry& registry,
                                              gl::StrSpan bytes) {
        int32_t total = 0, stripes = 0;
        for (unsigned index = 0; index < registry.stripeCount(); ++index) {
            const gl::BatchByteCandidates& stripe = registry.stripe(index);
            for (int32_t id = 1; id <= stripe.count(); ++id) {
                if (!(stripe.bytesAt(id) == bytes)) continue;
                total += stripe.nominationsAt(id);
                ++stripes;
            }
        }
        return { total, stripes };
    }
}

TEST(cold_hash_map, batch_dictionary_promotes_repeated_nominations_and_resolves_references) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), otherArena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::BatchByteDictionary dictionary(&global);
    dictionary.activate();
    gl::BatchByteCandidateRegistry candidates;
    candidates.init(&global, 4);
    candidates.init(&global, 4);
    ASSERT_EQ(candidates.stripeCount(), 4u);
    const std::string alpha(700, 'a'), beta(500, 'b'), gamma(300, 'g');
    nominateBytes(candidates, gl::StrSpan(alpha));
    nominateBytes(candidates, gl::StrSpan(beta));
    nominateBytes(candidates, gl::StrSpan(alpha));
    nominateBytes(candidates, gl::StrSpan(gamma));
    nominateBytes(candidates, gl::StrSpan(gamma));
    ASSERT_EQ(candidates.distinctCount(), 3);
    ASSERT_EQ(candidates.nominationTotal(), 5);
    ASSERT_TRUE(nominationsOf(candidates, gl::StrSpan(gamma)) == std::make_pair(2, 1));
    ASSERT_TRUE(nominationsOf(candidates, gl::StrSpan(beta)) == std::make_pair(1, 1));
    int64_t nominatedBytes = 0;
    for (unsigned index = 0; index < candidates.stripeCount(); ++index)
        nominatedBytes += candidates.stripe(index).liveBytes();
    ASSERT_TRUE(nominatedBytes > 0);
    ASSERT_EQ(dictionary.promoteNominations(candidates), 2);
    ASSERT_EQ(candidates.distinctCount(), 0);
    ASSERT_EQ(candidates.nominationTotal(), 0);
    for (unsigned index = 0; index < candidates.stripeCount(); ++index)
        ASSERT_EQ(candidates.stripe(index).liveBytes(), 0);
    ASSERT_EQ(dictionary.count(), 2);
    ASSERT_EQ(dictionary.logicalByteCount(), static_cast<int64_t>(alpha.size() + gamma.size()));
    ASSERT_TRUE(dictionary.liveBytes() >= dictionary.logicalByteCount());
    const int32_t alphaReference = dictionary.referenceOf(gl::StrSpan(alpha), gl::coldKeyHash(gl::StrSpan(alpha)));
    ASSERT_TRUE(alphaReference < 0);
    const gl::StrSpan alphaEntry = dictionary.bytesAt(alphaReference, static_cast<int32_t>(alpha.size()));
    ASSERT_TRUE(alphaEntry == gl::StrSpan(alpha));
    const int32_t gammaReference = dictionary.referenceOf(gl::StrSpan(gamma), gl::coldKeyHash(gl::StrSpan(gamma)));
    ASSERT_TRUE(gammaReference < 0 && gammaReference != alphaReference);
    const gl::StrSpan gammaEntry = dictionary.bytesAt(gammaReference, static_cast<int32_t>(gamma.size()));
    ASSERT_TRUE(gammaEntry == gl::StrSpan(gamma));
    ASSERT_EQ(dictionary.referenceOf(gl::StrSpan(beta), gl::coldKeyHash(gl::StrSpan(beta))), 0);

    using Bytes = gl::ColdHashSet<gl::BytesKeyStore>;
    Bytes shared(&arena, &dirty), oracle(&otherArena, &dirty);
    shared.keyStore().enableBatchSharing();
    for (const std::string* key : { &alpha, &beta, &gamma }) {
        shared.mint(gl::StrSpan(*key));
        oracle.mint(gl::StrSpan(*key));
    }
    shared.mint(gl::StrSpan());
    oracle.mint(gl::StrSpan());
    ASSERT_EQ(shared.keyAt(1).ptr, alphaEntry.ptr);
    ASSERT_EQ(shared.keyAt(3).ptr, gammaEntry.ptr);
    ASSERT_TRUE(shared.keyAt(2) == gl::StrSpan(beta));
    ASSERT_EQ(shared.decodeString(3), gamma);
    ASSERT_EQ(shared.lookup(gl::StrSpan(gamma)), 3);
    ASSERT_EQ(shared.lookup(gl::StrSpan()), 4);
    ASSERT_TRUE(canonicalBytes(shared) == canonicalBytes(oracle));
    ASSERT_EQ(shared.keyStore().hashStored(1), gl::coldKeyHash(gl::StrSpan(alpha)));
    shared.rebuildIndex();
    ASSERT_EQ(shared.lookup(gl::StrSpan(alpha)), 1);
    ASSERT_EQ(shared.keyStore().contentBytesFrom(0), oracle.keyStore().contentBytesFrom(0));

    Bytes copied(&otherArena, &dirty), reloaded(&arena, &dirty);
    copied.copyFrom(shared);
    ASSERT_EQ(copied.keyAt(1).ptr, alphaEntry.ptr);
    ASSERT_TRUE(canonicalBytes(copied) == canonicalBytes(oracle));
    reloaded.keyStore().enableBatchSharing();
    std::vector<char> lengths, bytes;
    oracle.appendLengthBytes(lengths, 0);
    oracle.appendContentBytes(bytes, 0);
    reloaded.stageLengths(lengths.data(), oracle.count());
    reloaded.consumeStagedLengths(bytes.data(), bytes.size());
    ASSERT_EQ(reloaded.keyAt(1).ptr, alphaEntry.ptr);
    ASSERT_EQ(reloaded.keyAt(3).ptr, shared.keyAt(3).ptr);
    ASSERT_TRUE(canonicalBytes(reloaded) == canonicalBytes(oracle));
    std::vector<char> tail;
    reloaded.appendContentBytesFromByte(tail, alpha.size());
    ASSERT_TRUE(tail == std::vector<char>(bytes.begin() + alpha.size(), bytes.end()));

    shared.release(); copied.release(); reloaded.release(); oracle.release();
}

TEST(cold_hash_map, batch_candidates_count_concurrent_nominations_once_per_payload) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{16 << 20, 1 << 18, 2048});
    gl::BatchByteCandidateRegistry candidates;
    candidates.init(&global, 8);
    std::vector<std::string> payloads;
    for (int i = 0; i < 50; ++i) payloads.push_back("payload/" + std::to_string(i) + std::string(40, 'p'));
    std::vector<std::thread> workers;
    for (int worker = 0; worker < 8; ++worker)
        workers.emplace_back([&candidates, &payloads]() {
            for (int round = 0; round < 20; ++round)
                for (const std::string& payload : payloads) nominateBytes(candidates, gl::StrSpan(payload));
        });
    for (std::thread& worker : workers) worker.join();
    ASSERT_EQ(candidates.distinctCount(), 50);
    ASSERT_EQ(candidates.nominationTotal(), 8000);
    for (const std::string& payload : payloads) {
        const gl::StrSpan bytes(payload);
        ASSERT_TRUE(nominationsOf(candidates, bytes) == std::make_pair(160, 1));
        // The stripe comes from digest bits the stripe's bucket index never uses.
        const unsigned expected = static_cast<unsigned>((gl::coldKeyHash(bytes) >> 32) % candidates.stripeCount());
        const gl::BatchByteCandidates& stripe = candidates.stripe(expected);
        bool held = false;
        for (int32_t id = 1; id <= stripe.count(); ++id) held = held || stripe.bytesAt(id) == bytes;
        ASSERT_TRUE(held);
    }
    candidates.releaseAll();
    ASSERT_EQ(candidates.distinctCount(), 0);
    ASSERT_EQ(candidates.nominationTotal(), 0);
}

TEST(cold_hash_map, batch_sharing_converts_pending_rows_compacts_and_nominates_once) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena firstArena(&global), secondArena(&global), oracleArena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::BatchByteDictionary dictionary(&global);
    dictionary.activate();
    gl::BatchByteCandidateRegistry candidates;
    candidates.init(&global, 2);
    using Bytes = gl::ColdHashSet<gl::BytesKeyStore>;
    Bytes first(&firstArena, &dirty), second(&secondArena, &dirty), oracle(&oracleArena, &dirty);
    first.keyStore().enableBatchSharing();
    second.keyStore().enableBatchSharing();
    const std::string a(900, 'a'), b(1500, 'b'), c(1200, 'c'), d(800, 'd'), e(400, 'e');
    for (const std::string* key : { &a, &b, &c }) { first.mint(gl::StrSpan(*key)); oracle.mint(gl::StrSpan(*key)); }
    for (const std::string* key : { &b, &c, &d }) second.mint(gl::StrSpan(*key));
    const auto nominate = [&](gl::StrSpan bytes, uint64_t hash) { candidates.nominate(bytes, hash); };
    ASSERT_EQ(first.keyStore().sharePendingRows(), 0);
    ASSERT_EQ(first.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(second.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(first.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(candidates.distinctCount(), 4);
    ASSERT_TRUE(nominationsOf(candidates, gl::StrSpan(a)) == std::make_pair(1, 1));
    ASSERT_TRUE(nominationsOf(candidates, gl::StrSpan(b)) == std::make_pair(2, 1));
    ASSERT_EQ(dictionary.promoteNominations(candidates), 2);
    const int64_t before = first.liveBytes();
    ASSERT_EQ(first.keyStore().sharePendingRows(), 2);
    first.keyStore().compactAfterBatchConversion(scratch);
    ASSERT_EQ(first.keyStore().sharePendingRows(), 0);
    ASSERT_EQ(second.keyStore().sharePendingRows(), 2);
    second.keyStore().compactAfterBatchConversion(scratch);
    ASSERT_EQ(first.keyAt(2).ptr, second.keyAt(1).ptr);
    ASSERT_EQ(first.keyAt(3).ptr, second.keyAt(2).ptr);
    ASSERT_TRUE(first.liveBytes() < before);
    ASSERT_TRUE(canonicalBytes(first) == canonicalBytes(oracle));
    ASSERT_EQ(first.decodeString(1), a);
    ASSERT_EQ(first.lookup(gl::StrSpan(a)), 1);
    ASSERT_EQ(second.decodeString(3), d);

    first.mint(gl::StrSpan(e)); oracle.mint(gl::StrSpan(e));
    first.mint(gl::StrSpan(c));
    ASSERT_EQ(first.count(), 4);
    Bytes third(&secondArena, &dirty);
    third.keyStore().enableBatchSharing();
    third.mint(gl::StrSpan(c));
    ASSERT_EQ(third.keyAt(1).ptr, first.keyAt(3).ptr);
    ASSERT_EQ(first.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(candidates.distinctCount(), 1);
    ASSERT_TRUE(nominationsOf(candidates, gl::StrSpan(e)) == std::make_pair(1, 1));
    first.keyStore().compactAfterBatchConversion(scratch);
    ASSERT_TRUE(canonicalBytes(first) == canonicalBytes(oracle));
    candidates.releaseAll();

    first.release(); second.release(); third.release(); oracle.release();
}

TEST(cold_hash_map, batch_sharing_marks_follow_stable_erase) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::BatchByteDictionary dictionary(&global);
    dictionary.activate();
    gl::BatchByteCandidateRegistry candidates;
    candidates.init(&global, 1);
    gl::ColdHashSet<gl::BytesKeyStore> keys(&arena, &dirty);
    keys.keyStore().enableBatchSharing();
    const auto key = [](int i) { return std::string(static_cast<std::size_t>(64 + i), static_cast<char>(97 + i)); };
    for (int i = 0; i < 6; ++i) keys.mint(gl::StrSpan(key(i)));
    const auto nominate = [&](gl::StrSpan bytes, uint64_t hash) { candidates.nominate(bytes, hash); };
    ASSERT_EQ(keys.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(candidates.distinctCount(), 6);
    candidates.releaseAll();
    ASSERT_EQ(keys.eraseIf([&](gl::StrSpan bytes) { return bytes.len == 65 || bytes.len == 68; }), 2);
    ASSERT_EQ(keys.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(candidates.distinctCount(), 0);
    keys.mint(gl::StrSpan(key(7)));
    ASSERT_EQ(keys.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(candidates.distinctCount(), 1);
    ASSERT_TRUE(nominationsOf(candidates, gl::StrSpan(key(7))) == std::make_pair(1, 1));
    nominateBytes(candidates, gl::StrSpan(key(7)));
    ASSERT_EQ(dictionary.promoteNominations(candidates), 1);
    ASSERT_EQ(keys.keyStore().sharePendingRows(), 1);
    keys.keyStore().compactAfterBatchConversion(scratch);
    ASSERT_EQ(keys.decodeString(5), key(7));
    ASSERT_EQ(keys.decodeString(4), key(5));
    keys.resetToFresh();
    keys.mint(gl::StrSpan(key(2)));
    ASSERT_EQ(keys.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(candidates.distinctCount(), 1);
    candidates.releaseAll();
    keys.release();
}

TEST(cold_hash_map, batch_sharing_repacks_prefix_packed_local_rows) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{32 << 20, 1 << 18, 2048});
    gl::LbArena arena(&global), otherArena(&global), oracleArena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::BatchByteDictionary dictionary(&global);
    dictionary.activate();
    gl::BatchByteCandidateRegistry candidates;
    candidates.init(&global, 4);
    gl::PrefixKeyMetadata prefixes(&arena, &dirty);
    gl::ColdStringTable packed(&arena, &dirty), peer(&otherArena, &dirty), oracle(&oracleArena, &dirty);
    packed.keyStore().bindPrefixes(prefixes);
    packed.keyStore().enableBatchSharing();
    peer.keyStore().enableBatchSharing();
    std::vector<std::string> keys;
    for (int i = 0; i < 1500; ++i) keys.push_back("origin/history/line/" + std::to_string(100000 + 7 * i));
    for (const std::string& k : keys) { packed.intern(gl::StrSpan(k)); oracle.intern(gl::StrSpan(k)); }
    for (int i = 0; i < 1500; i += 3) peer.intern(gl::StrSpan(keys[i]));
    packed.keyStore().packPrefixes(scratch);
    const int64_t packedBytes = packed.liveBytes();
    ASSERT_TRUE(canonicalBytes(packed) == canonicalBytes(oracle));
    const auto nominate = [&](gl::StrSpan bytes, uint64_t hash) { candidates.nominate(bytes, hash); };
    ASSERT_EQ(packed.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(peer.keyStore().shareNewRows(nominate), 0);
    ASSERT_EQ(dictionary.promoteNominations(candidates), 500);
    ASSERT_EQ(packed.keyStore().sharePendingRows(), 500);
    packed.keyStore().compactAfterBatchConversion(scratch);
    ASSERT_EQ(peer.keyStore().sharePendingRows(), 500);
    const gl::StrSpan firstKey(keys[0]);
    ASSERT_EQ(peer.view(1).ptr, dictionary.bytesAt(dictionary.referenceOf(firstKey, gl::coldKeyHash(firstKey)),
                                                  static_cast<int32_t>(keys[0].size())).ptr);
    ASSERT_TRUE(packed.liveBytes() < packedBytes);
    ASSERT_TRUE(canonicalBytes(packed) == canonicalBytes(oracle));
    gl::LbArena copyScratch(&global);
    for (int id = 1; id <= 1500; ++id) {
        ASSERT_EQ(packed.decodeString(id), keys[id - 1]);
        ASSERT_TRUE(packed.keyStore().copyView(id, copyScratch) == gl::StrSpan(keys[id - 1]));
    }
    copyScratch.releaseAll();
    for (int i = 0; i < 1500; i += 250) ASSERT_EQ(packed.lookup(gl::StrSpan(keys[i])), i + 1);
    ASSERT_EQ(packed.keyStore().compareKeys(1, 2) < 0, keys[0] < keys[1]);
    gl::ColdStringTable copy(&otherArena, &dirty);
    copy.copyFrom(packed);
    ASSERT_TRUE(canonicalBytes(copy) == canonicalBytes(oracle));
    packed.release(); peer.release(); oracle.release(); copy.release();
}

TEST(cold_hash_map, shared_native_pool_payload_rows_take_part_in_batch_sharing) {
    gl::GlobalMemoryManager global;
    global.init(gl::StaticMemoryConfig{8 << 20, 1 << 18, 2048});
    gl::LbArena firstArena(&global), secondArena(&global), scratch(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::BatchByteDictionary dictionary(&global);
    dictionary.activate();
    gl::BatchByteCandidateRegistry candidates;
    candidates.init(&global, 2);
    gl::SharedBytePool firstPool(&firstArena, &dirty), secondPool(&secondArena, &dirty);
    firstPool.payloadStore().enableBatchSharing();
    secondPool.payloadStore().enableBatchSharing();
    using Bytes = gl::ColdHashSet<gl::BytesKeyStore>;
    Bytes firstKeys(&firstArena, &dirty), secondKeys(&secondArena, &dirty);
    firstKeys.keyStore().bindSharedBytes(firstPool);
    secondKeys.keyStore().bindSharedBytes(secondPool);
    const std::string rule(1100, 'r'), dead(900, 'x');
    firstKeys.mint(gl::StrSpan(dead));
    firstKeys.mint(gl::StrSpan(rule));
    secondKeys.mint(gl::StrSpan(rule));
    const auto nominate = [&](gl::StrSpan bytes, uint64_t hash) { candidates.nominate(bytes, hash); };
    ASSERT_EQ(firstPool.payloadStore().shareNewRows(nominate), 0);
    ASSERT_EQ(secondPool.payloadStore().shareNewRows(nominate), 0);
    ASSERT_EQ(dictionary.promoteNominations(candidates), 1);
    ASSERT_EQ(firstPool.payloadStore().sharePendingRows(), 1);
    firstPool.payloadStore().compactAfterBatchConversion(scratch);
    ASSERT_EQ(secondPool.payloadStore().sharePendingRows(), 1);
    secondPool.payloadStore().compactAfterBatchConversion(scratch);
    ASSERT_EQ(firstKeys.keyAt(2).ptr, secondKeys.keyAt(1).ptr);
    ASSERT_EQ(firstKeys.lookup(gl::StrSpan(rule)), 2);
    ASSERT_EQ(firstKeys.eraseIf([&](gl::StrSpan key) { return key.len == static_cast<int>(dead.size()); }), 1);
    firstPool.collect(scratch);
    ASSERT_EQ(firstPool.count(), 1);
    ASSERT_EQ(firstKeys.decodeString(1), rule);
    ASSERT_EQ(firstPool.payloadStore().shareNewRows(nominate), 0);
    ASSERT_EQ(candidates.distinctCount(), 0);
    firstKeys.release(); secondKeys.release();
    firstPool.release(); secondPool.release();
}

TEST(cold_hash_map, mint_hashed_matches_mint) {
    gl::GlobalMemoryManager global;
    global.init(kMapTestCfg);
    gl::LbArena arena(&global), otherArena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::BytesKeyStore> plain(&arena, &dirty), hashed(&otherArena, &dirty);
    for (int i = 0; i < 600; ++i) {
        const std::string key = "key/" + std::to_string(i % 250);
        const gl::StrSpan span(key);
        ASSERT_EQ(hashed.mintHashed(span, gl::coldKeyHash(span)), plain.mint(span));
    }
    ASSERT_EQ(hashed.count(), 250);
    for (int i = 0; i < 260; ++i) {
        const std::string key = "key/" + std::to_string(i);
        ASSERT_EQ(hashed.lookup(gl::StrSpan(key)), plain.lookup(gl::StrSpan(key)));
    }
    PodSet podPlain(&arena, &dirty), podHashed(&otherArena, &dirty);
    for (int32_t i = 0; i < 90; ++i) {
        const int32_t key = (i * 7) % 40;
        ASSERT_EQ(podHashed.mintHashed(key, podHashed.keyStore().hashProbe(key)), podPlain.mint(key));
    }
    ASSERT_EQ(podHashed.count(), 40);
    ASSERT_EQ(podHashed.lookup(13), podPlain.lookup(13));
}

TEST(cold_hash_map, lookup_hashed_matches_lookup) {
    gl::GlobalMemoryManager global;
    global.init(kMapTestCfg);
    gl::LbArena arena(&global);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdStringTable table(&arena, &dirty);
    ASSERT_EQ(table.lookupHashed(gl::StrSpan("k", 1), gl::coldKeyHash(gl::StrSpan("k", 1))), 0);
    for (int i = 0; i < 200; ++i) table.intern(gl::StrSpan(std::to_string(i)));
    for (int i = 0; i < 220; ++i) {
        const std::string key = std::to_string(i);
        ASSERT_EQ(table.lookupHashed(gl::StrSpan(key), gl::coldKeyHash(gl::StrSpan(key))),
                  table.lookup(gl::StrSpan(key)));
    }
    PodSet ids(&arena, &dirty);
    for (int32_t i = 0; i < 10; ++i) ids.mint(i);
    ASSERT_EQ(ids.eraseIf([](int32_t id) { return (id & 1) != 0; }), 5);
    ASSERT_EQ(ids.lookup(4), 3);
}
