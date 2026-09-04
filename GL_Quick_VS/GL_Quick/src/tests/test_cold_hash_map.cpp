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

#include <string>
#include <type_traits>
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
