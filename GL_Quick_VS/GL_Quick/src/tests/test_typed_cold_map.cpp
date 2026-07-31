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
/// @brief Unit tests for the `TypedCold*` typed-façade layer + the `Codec<K>`
///        key codecs.
///
/// @details
/// Proves the three load-bearing properties the façade rests on:
///   1. Each key `Codec` reproduces the exact byte/scalar layout of the
///      hand-rolled helper it replaces (`encodeEqClassKey`, `packStatementKey`,
///      `packLbStateKey`) and round-trips `decode(encode(k)) == k`.
///   2. A `TypedCold*` wrapper produces deload columns BYTE-IDENTICAL to the raw
///      engine map built with the same logical content — the wrapper adds a typed
///      surface and zero stored bytes.
///   3. The re-exported deload facet types bind to `inner()` and stream / reload
///      exactly as `LbMemory` will wire them.
/// Plus the typed `eraseIf` / `eraseSetIf` predicate, `copyFrom`, and a blob-map
/// record round trip through a test `Codec<Record>`.

#include "test_harness.hpp"

#include "../memory_infra/typed_cold_map.hpp"

#include <cstdint>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block — the bump-arena substrate this branch uses.
    const gl::StaticMemoryConfig kTypedCfg{ 1 << 20, 1 << 18 };

    // The byte layout encodeEqClassKey produced: NameId LE validity then members.
    inline std::string refEqClassBytes(gl::NameId validity,
                                       const std::vector<gl::NameId>& members) {
        std::string key(sizeof(gl::NameId) * (members.size() + 1), '\0');
        std::memcpy(&key[0], &validity, sizeof(gl::NameId));
        for (std::size_t i = 0; i < members.size(); ++i)
            std::memcpy(&key[sizeof(gl::NameId) * (i + 1)], &members[i],
                        sizeof(gl::NameId));
        return key;
    }

    // A test record with a variable-length field — the blob-map value shape.
    struct TestRecord {
        int32_t a;
        std::vector<int32_t> xs;
        bool operator==(const TestRecord& o) const {
            return a == o.a && xs == o.xs;
        }
    };
}

namespace gl {
    // A test value codec (record serialization): int32 a, int32 count, ids.
    template <>
    struct Codec<TestRecord> {
        static std::vector<char> serialize(const TestRecord& r) {
            std::vector<char> out;
            const int32_t n = static_cast<int32_t>(r.xs.size());
            const char* pa = reinterpret_cast<const char*>(&r.a);
            const char* pn = reinterpret_cast<const char*>(&n);
            out.insert(out.end(), pa, pa + sizeof(int32_t));
            out.insert(out.end(), pn, pn + sizeof(int32_t));
            for (int32_t x : r.xs) {
                const char* px = reinterpret_cast<const char*>(&x);
                out.insert(out.end(), px, px + sizeof(int32_t));
            }
            return out;
        }
        static TestRecord deserialize(const char* p, int32_t n) {
            TestRecord r{};
            int32_t cur = 0;
            std::memcpy(&r.a, p + cur, sizeof(int32_t));
            cur += sizeof(int32_t);
            int32_t count = 0;
            std::memcpy(&count, p + cur, sizeof(int32_t));
            cur += sizeof(int32_t);
            r.xs.resize(static_cast<std::size_t>(count));
            for (int32_t i = 0; i < count; ++i) {
                std::memcpy(&r.xs[static_cast<std::size_t>(i)], p + cur,
                            sizeof(int32_t));
                cur += sizeof(int32_t);
            }
            assert(cur == n && "TestRecord codec round-trip length mismatch");
            return r;
        }
    };
}

// ---- 1. Codec formula match + round trip -----------------------------------

TEST(typed_cold_map, codec_eqclasskey_layout_and_round_trip) {
    using gl::EqClassKey;
    const std::vector<std::vector<gl::NameId>> memberSets = {
        {}, { 5 }, { -3, 7, 1000 }, { 0, 1, 2, 3, 4 }, { 70000, 200000 } };
    for (gl::NameId v : { gl::NameId(0), gl::NameId(7), gl::NameId(-9),
                          gl::NameId(32000), gl::NameId(1000000) }) {
        for (const auto& members : memberSets) {
            const EqClassKey k{ v, members };
            const std::string enc = gl::Codec<EqClassKey>::encode(k);
            ASSERT_TRUE(enc == refEqClassBytes(v, members));   // matches legacy
            const EqClassKey back =
                gl::Codec<EqClassKey>::decode(gl::StrSpan(enc));
            ASSERT_TRUE(back == k);                            // round trip
        }
    }
}

TEST(typed_cold_map, codec_statementkey_pack_unpack) {
    using gl::StatementKey;
    // Include values past the old 16-bit ceiling to prove the halves no longer
    // alias (the whole point of the migration).
    const gl::NameId corners[] = { 0, 1, -1, 32000, -32000, 12345, -9999,
                                   70000, 200000, 1000000, -1000000 };
    for (gl::NameId o : corners) {
        for (gl::NameId v : corners) {
            const StatementKey k{ o, v };
            const int64_t packed = gl::Codec<StatementKey>::encode(k);
            // The packStatementKey formula: packInt32Pair(orig, valid) — orig
            // in the high 32 bits, valid in the low 32.
            const int64_t ref = gl::packInt32Pair(o, v);
            ASSERT_EQ(packed, ref);
            // Halves recover via >>32 / &0xFFFFFFFF.
            ASSERT_EQ(static_cast<uint32_t>(static_cast<uint64_t>(packed) >> 32),
                      static_cast<uint32_t>(o));
            ASSERT_EQ(static_cast<uint32_t>(
                          static_cast<uint64_t>(packed) & 0xFFFFFFFFu),
                      static_cast<uint32_t>(v));
            const StatementKey back = gl::Codec<StatementKey>::decode(packed);
            ASSERT_TRUE(back == k);
        }
    }
}

TEST(typed_cold_map, codec_lbstatepairkey_pack_unpack) {
    using gl::LbStatePairKey;
    const int32_t corners[] = { 0, 1, -1, 2000000000, -2000000000, 123456 };
    for (int32_t h : corners) {
        for (int32_t l : corners) {
            const LbStatePairKey k{ h, l };
            const int64_t packed = gl::Codec<LbStatePairKey>::encode(k);
            const int64_t ref = static_cast<int64_t>(
                (static_cast<uint64_t>(static_cast<uint32_t>(h)) << 32)
                | static_cast<uint64_t>(static_cast<uint32_t>(l)));
            ASSERT_EQ(packed, ref);
            const LbStatePairKey back = gl::Codec<LbStatePairKey>::decode(packed);
            ASSERT_TRUE(back == k);
        }
    }
}

TEST(typed_cold_map, pack_int32_pair) {
    using gl::packInt32Pair;
    const int32_t corners[] = { 0, 1, -1, 2000000000, -2000000000, 123456 };
    for (int32_t hi : corners) {
        for (int32_t lo : corners) {
            const int64_t p = packInt32Pair(hi, lo);
            // Unpacking each half (unsigned view) recovers the original id,
            // for negative sentinels too — the property the round-trip needs.
            ASSERT_EQ(static_cast<uint32_t>(static_cast<uint64_t>(p) >> 32),
                      static_cast<uint32_t>(hi));
            ASSERT_EQ(static_cast<uint32_t>(
                          static_cast<uint64_t>(p) & 0xFFFFFFFFu),
                      static_cast<uint32_t>(lo));
        }
    }
    // Bijection sanity: distinct pairs pack distinctly, order matters.
    ASSERT_TRUE(packInt32Pair(-1, 0) != packInt32Pair(0, -1));
    ASSERT_TRUE(packInt32Pair(5, 7) != packInt32Pair(7, 5));
}

TEST(typed_cold_map, codec_normkey_round_trip) {
    using gl::NormKey;
    const std::vector<std::vector<gl::NameId>> dataSets = {
        {}, { 5 }, { -3, 7, 1000 }, { 0, 1, 2, 3, 4 }, { 70000, 1000000 } };
    for (int32_t ne : { 0, 1, 7, 32000, 1000000 }) {
        for (const auto& data : dataSets) {
            const NormKey k{ ne, data };
            const std::string enc = gl::Codec<NormKey>::encode(k);
            ASSERT_EQ(static_cast<int>(enc.size()),
                      static_cast<int>(sizeof(gl::NameId) * (data.size() + 2)));
            const NormKey back = gl::Codec<NormKey>::decode(gl::StrSpan(enc));
            ASSERT_TRUE(back == k);
        }
    }
}

TEST(typed_cold_map, codec_int16setkey_round_trip) {
    using gl::Int16SetKey;
    const std::vector<std::vector<gl::NameId>> idSets = {
        {}, { 1 }, { -5, 0, 5 }, { -32000, 0, 32000 }, { 0, 70000, 1000000 } };
    for (const auto& ids : idSets) {
        const Int16SetKey k{ ids };
        const std::string enc = gl::Codec<Int16SetKey>::encode(k);
        ASSERT_EQ(static_cast<int>(enc.size()),
                  static_cast<int>(sizeof(gl::NameId) * (ids.size() + 1)));
        const Int16SetKey back =
            gl::Codec<Int16SetKey>::decode(gl::StrSpan(enc));
        ASSERT_TRUE(back == k);
    }
}

TEST(typed_cold_map, codec_idveckey_round_trip) {
    using gl::IdVecKey;
    const std::vector<std::vector<int32_t>> idSets = {
        {}, { 7 }, { -3, 0, 1000000 }, { 1, 2, 3, 4, 5 } };
    for (const auto& ids : idSets) {
        const IdVecKey k{ ids };
        const std::string enc = gl::Codec<IdVecKey>::encode(k);
        ASSERT_EQ(static_cast<int>(enc.size()),
                  static_cast<int>(sizeof(int32_t) * (ids.size() + 1)));
        const IdVecKey back = gl::Codec<IdVecKey>::decode(gl::StrSpan(enc));
        ASSERT_TRUE(back == k);
    }
}

// The NameId alias is a signed 32-bit integer; MAX_NAME_IDS is a NameId raised
// to 1,000,000 now that the 32-bit width admits ids far past the old ceiling.
TEST(typed_cold_map, nameid_is_32bit_signed) {
    static_assert(std::is_signed_v<gl::NameId>, "NameId must be signed");
    static_assert(sizeof(gl::NameId) >= 4, "NameId must be at least 32-bit");
    static_assert(std::is_same_v<gl::NameId, std::int32_t>,
                  "NameId is int32_t");
    static_assert(std::is_same_v<
                      std::remove_const_t<decltype(gl::ExecutionParameters::MAX_NAME_IDS)>,
                      gl::NameId>,
                  "MAX_NAME_IDS must be a NameId");
    ASSERT_EQ(static_cast<int>(gl::ExecutionParameters::MAX_NAME_IDS), 1000000);
}

// ---- 2. Byte-identity: typed wrapper vs the raw engine map -----------------

TEST(typed_cold_map, map_byte_identity_vs_raw_byte_key) {
    // TypedColdMap<EqClassKey,int> over a BytesKeyStore vs the raw byte-key map
    // fed the SAME bytes through encodeEqClassKey's layout.
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lbT(&g);
    gl::DirtyState dT = gl::DirtyState::Clean;
    gl::TypedColdMap<gl::EqClassKey, int> typed(&lbT, &dT);

    gl::LbArena lbR(&g);
    gl::DirtyState dR = gl::DirtyState::Clean;
    gl::ColdHashMap<gl::BytesKeyStore, int> raw(&lbR, &dR);

    for (gl::NameId v = 1; v <= 40; ++v) {
        std::vector<gl::NameId> members{ v, gl::NameId(v * 3), gl::NameId(v - 7) };
        typed.insert(gl::EqClassKey{ v, members }, v * 100);
        const std::string b = refEqClassBytes(v, members);
        raw.insert(gl::StrSpan(b), v * 100);
    }
    ASSERT_EQ(typed.count(), raw.count());

    std::vector<char> lt, bt, vt, lr, br, vr;
    typed.inner().appendLengthBytes(lt, 0);
    typed.inner().appendContentBytes(bt, 0);
    typed.inner().appendValueBytes(vt, 0);
    raw.appendLengthBytes(lr, 0);
    raw.appendContentBytes(br, 0);
    raw.appendValueBytes(vr, 0);
    ASSERT_TRUE(lt == lr);
    ASSERT_TRUE(bt == br);
    ASSERT_TRUE(vt == vr);

    // Typed read surface agrees with the stored content.
    const int* p = typed.find(gl::EqClassKey{ 5, { 5, 15, -2 } });
    ASSERT_TRUE(p != nullptr && *p == 500);
    ASSERT_EQ(typed.findOr(gl::EqClassKey{ 999, {} }, 0), 0);   // miss
}

TEST(typed_cold_map, map_byte_identity_vs_raw_packed_key) {
    // TypedColdMap<StatementKey,int> over a PodKeyStore<int64> vs the raw map fed
    // the same packed scalars (StatementKey packs to int64 now).
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lbT(&g);
    gl::DirtyState dT = gl::DirtyState::Clean;
    gl::TypedColdMap<gl::StatementKey, int> typed(&lbT, &dT);

    gl::LbArena lbR(&g);
    gl::DirtyState dR = gl::DirtyState::Clean;
    gl::ColdHashMap<gl::PodKeyStore<int64_t>, int> raw(&lbR, &dR);

    for (int16_t i = 0; i < 60; ++i) {
        const gl::StatementKey k{ int16_t(i * 7 + 1), int16_t(i + 2) };
        typed.insert(k, i + 1);
        raw.insert(gl::Codec<gl::StatementKey>::encode(k), i + 1);
    }
    std::vector<char> kt, vt, kr, vr;
    typed.inner().keyStore().appendKeyBytes(kt, 0);
    typed.inner().appendValueBytes(vt, 0);
    raw.keyStore().appendKeyBytes(kr, 0);
    raw.appendValueBytes(vr, 0);
    ASSERT_TRUE(kt == kr);
    ASSERT_TRUE(vt == vr);

    // The raw decode/keyAt passthroughs (the sacred-dump accessors) return the
    // packed scalar, equal to the raw map's at the same id; decodeKey returns the
    // typed key (id 1 is the first inserted key {1,2}).
    ASSERT_EQ(typed.decode(1), raw.decode(1));
    ASSERT_EQ(typed.keyAt(1), raw.keyAt(1));
    ASSERT_TRUE((typed.decodeKey(1) == gl::StatementKey{ 1, 2 }));

    // upsert: overwrite on hit, insert on miss.
    typed.upsert(gl::StatementKey{ 1, 2 }, 777);
    ASSERT_EQ(*typed.find(gl::StatementKey{ 1, 2 }), 777);
    typed.upsert(gl::StatementKey{ 9999, 1 }, 5);
    ASSERT_EQ(*typed.find(gl::StatementKey{ 9999, 1 }), 5);
}

TEST(typed_cold_map, set_value_at_relaxed_updates_without_dirtying) {
    // The mail-cursor advance door: an in-place value update on a PRE-CREATED
    // key that leaves the shared dirty flag untouched — the property that makes
    // the parallel disjoint cursor advance race-free on the never-deloaded mail
    // pool (no thread writes the one *dirty_ byte).
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdMap<gl::StatementKey, int> m(&lb, &d);
    // Pre-create the cells single-threaded (the registration analogue).
    for (int16_t i = 0; i < 8; ++i)
        m.insert(gl::StatementKey{ int16_t(i + 1), int16_t(0) }, 0);
    d = gl::DirtyState::Clean;                            // observe the next write
    m.setValueAtRelaxed(gl::StatementKey{ int16_t(3), int16_t(0) }, 42);
    m.setValueAtRelaxed(gl::StatementKey{ int16_t(8), int16_t(0) }, 99);
    ASSERT_EQ(*m.find(gl::StatementKey{ int16_t(3), int16_t(0) }), 42);  // updated in place
    ASSERT_EQ(*m.find(gl::StatementKey{ int16_t(8), int16_t(0) }), 99);
    ASSERT_EQ(*m.find(gl::StatementKey{ int16_t(1), int16_t(0) }), 0);   // neighbour untouched
    ASSERT_EQ(static_cast<int>(d),
              static_cast<int>(gl::DirtyState::Clean));   // NOT escalated
}

TEST(typed_cold_map, set_byte_identity_and_typed_erase) {
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lbT(&g);
    gl::DirtyState dT = gl::DirtyState::Clean;
    gl::TypedColdSet<gl::LbStatePairKey> typed(&lbT, &dT);

    gl::LbArena lbR(&g);
    gl::DirtyState dR = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::PodKeyStore<int64_t>> raw(&lbR, &dR);

    for (int32_t i = 0; i < 50; ++i) {
        const gl::LbStatePairKey k{ i + 1, i * 2 };
        typed.mint(k);
        raw.mint(gl::Codec<gl::LbStatePairKey>::encode(k));
    }
    std::vector<char> kt, kr;
    typed.inner().keyStore().appendKeyBytes(kt, 0);
    raw.keyStore().appendKeyBytes(kr, 0);
    ASSERT_TRUE(kt == kr);

    // Typed predicate erase: drop every key whose high field is even.
    const int32_t removed = typed.eraseIf(
        [](const gl::LbStatePairKey& k) { return (k.high % 2) == 0; });
    ASSERT_TRUE(removed > 0);
    ASSERT_FALSE(typed.contains(gl::LbStatePairKey{ 2, 2 }));   // high=2 erased
    ASSERT_TRUE(typed.contains(gl::LbStatePairKey{ 1, 0 }));    // high=1 kept
}

TEST(typed_cold_map, set_map_byte_identity_and_comparator) {
    // The orBookkeeping shape: a stateful per-call comparator forwarded through.
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lbT(&g);
    gl::DirtyState dT = gl::DirtyState::Clean;
    gl::TypedColdSetMap<gl::StatementKey, int32_t> typed(&lbT, &dT);

    gl::LbArena lbR(&g);
    gl::DirtyState dR = gl::DirtyState::Clean;
    gl::ColdSetMap<gl::PodKeyStore<int64_t>, int32_t> raw(&lbR, &dR);

    // rank[v] = -v: a comparator that sorts the run by descending value id.
    int32_t rank[64];
    for (int32_t i = 0; i < 64; ++i) rank[i] = -i;
    struct ByRank {
        const int32_t* rank;
        bool operator()(int32_t a, int32_t b) const {
            return rank[a] < rank[b];
        }
    };
    const ByRank cmp{ rank };

    for (int16_t i = 0; i < 20; ++i) {
        const gl::StatementKey k{ int16_t(i + 1), int16_t(i + 100) };
        const int64_t packed = gl::Codec<gl::StatementKey>::encode(k);
        for (int32_t v : { 3, 1, 2, 1 }) {            // includes a dup
            typed.insertSorted(k, (v + i) % 64, cmp);
            raw.insertSorted(packed, (v + i) % 64, cmp);
        }
    }
    std::vector<char> kt, rt, vt, kr, rr, vr;
    typed.inner().keyStore().appendKeyBytes(kt, 0);
    typed.inner().appendRunStartBytes(rt, 0);
    typed.inner().appendValueBytes(vt, 0);
    raw.keyStore().appendKeyBytes(kr, 0);
    raw.appendRunStartBytes(rr, 0);
    raw.appendValueBytes(vr, 0);
    ASSERT_TRUE(kt == kr);
    ASSERT_TRUE(rt == rr);
    ASSERT_TRUE(vt == vr);
}

TEST(typed_cold_map, blob_map_record_round_trip_and_identity) {
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lbT(&g);
    gl::DirtyState dT = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int16_t, TestRecord> typed(&lbT, &dT);

    gl::LbArena lbR(&g);
    gl::DirtyState dR = gl::DirtyState::Clean;
    gl::ColdBlobMap<gl::PodKeyStore<int16_t>> raw(&lbR, &dR);

    for (int16_t k = 1; k <= 15; ++k) {
        std::vector<TestRecord> recs = {
            TestRecord{ k, { k, int32_t(k * 2) } },
            TestRecord{ int32_t(k + 100), {} } };
        typed.assignRun(k, recs);
        // Raw: serialize the same records by hand and assignRun.
        std::vector<char> bytes;
        std::vector<int32_t> lens;
        for (const TestRecord& r : recs) {
            const std::vector<char> b = gl::Codec<TestRecord>::serialize(r);
            lens.push_back(static_cast<int32_t>(b.size()));
            bytes.insert(bytes.end(), b.begin(), b.end());
        }
        raw.assignRun(k, bytes.data(), lens.data(),
                      static_cast<int32_t>(recs.size()));
    }

    // Record round trip through the typed surface.
    const int32_t id = typed.lookup(7);
    ASSERT_TRUE(id != 0);
    ASSERT_EQ(typed.runLen(id), 2);
    const TestRecord r0 = typed.recordAt(id, 0);
    ASSERT_TRUE((r0 == TestRecord{ 7, { 7, 14 } }));
    const std::vector<TestRecord> all = typed.recordsAt(id);
    ASSERT_EQ(static_cast<int32_t>(all.size()), 2);

    // Byte-identity of all four blob columns.
    std::vector<char> kt, rt, bt, pt, kr, rr, br, pr;
    typed.inner().keyStore().appendKeyBytes(kt, 0);
    typed.inner().appendRunStartBytes(rt, 0);
    typed.inner().appendBlobStartBytes(bt, 0);
    typed.inner().appendBlobPoolBytes(pt, 0);
    raw.keyStore().appendKeyBytes(kr, 0);
    raw.appendRunStartBytes(rr, 0);
    raw.appendBlobStartBytes(br, 0);
    raw.appendBlobPoolBytes(pr, 0);
    ASSERT_TRUE(kt == kr);
    ASSERT_TRUE(rt == rr);
    ASSERT_TRUE(bt == br);
    ASSERT_TRUE(pt == pr);
}

TEST(typed_cold_map, int64_key_identity_and_blob_map) {
    // Codec<int64_t> is the identity key codec for the cold exprOriginMap (packed
    // (expressionId, validityId) int64 keys). Round-trip the codec, then store and
    // retrieve a run through a blob map keyed on int64 end-to-end.
    const int64_t k = 0x0123456789ABCDEFLL;
    ASSERT_TRUE(gl::Codec<int64_t>::encode(k) == k);
    ASSERT_TRUE(gl::Codec<int64_t>::decode(k) == k);

    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int64_t, TestRecord> m(&lb, &d);

    const int64_t key = 0x1111222233334444LL;
    const std::vector<TestRecord> recs = {
        TestRecord{ 7, { 1, 2, 3 } }, TestRecord{ 8, {} } };
    m.assignRun(key, recs);
    const int32_t id = m.lookup(key);
    ASSERT_TRUE(id != 0);
    ASSERT_EQ(m.runLen(id), 2);
    ASSERT_TRUE((m.recordAt(id, 0) == TestRecord{ 7, { 1, 2, 3 } }));
    const std::vector<TestRecord> all = m.recordsAt(id);
    ASSERT_EQ(static_cast<int32_t>(all.size()), 2);
    ASSERT_EQ(m.lookup(0x5555555555555555LL), 0);   // absent key probes to 0
}

TEST(typed_cold_map, blob_append_record_matches_assignrun) {
    // appendRecord(k, r) (the encodedMap RMW fast path) must leave the store
    // BYTE-IDENTICAL to building the same run with one assignRun(k, initial+[r]).
    // Covers the existing-key splice (HashMap::appendBlobToRun, incl. the
    // later-key run-start rebase + byte-tail shift when the grown key is NOT the
    // pool tail) AND the brand-new-key branch (lookup miss -> one-blob assignRun).
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);

    // viaAppend: create keys 1..12 with a two-record run, then append one more
    // record to each in ascending order -- so key 1 grows while keys 2..12 sit
    // after it, exercising the rebase. Key 13 is never assignRun'd; appendRecord
    // must mint it.
    gl::LbArena lbA(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int16_t, TestRecord> viaAppend(&lbA, &dA);
    for (int16_t k = 1; k <= 12; ++k)
        viaAppend.assignRun(k, std::vector<TestRecord>{
            TestRecord{ k, { k, int32_t(k * 2) } },
            TestRecord{ int32_t(k + 100), {} } });
    for (int16_t k = 1; k <= 12; ++k)
        viaAppend.appendRecord(
            k, TestRecord{ int32_t(k + 500), { int32_t(k * 3) } });
    viaAppend.appendRecord(int16_t(13), TestRecord{ 1313, { 13 } });

    // viaAssign: the same content, each run built in a single assignRun.
    gl::LbArena lbB(&g);
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int16_t, TestRecord> viaAssign(&lbB, &dB);
    for (int16_t k = 1; k <= 12; ++k)
        viaAssign.assignRun(k, std::vector<TestRecord>{
            TestRecord{ k, { k, int32_t(k * 2) } },
            TestRecord{ int32_t(k + 100), {} },
            TestRecord{ int32_t(k + 500), { int32_t(k * 3) } } });
    viaAssign.assignRun(int16_t(13),
                        std::vector<TestRecord>{ TestRecord{ 1313, { 13 } } });

    // Byte-identity of all four blob columns.
    std::vector<char> ka, ra, ba, pa, kb, rb, bb, pb;
    viaAppend.inner().keyStore().appendKeyBytes(ka, 0);
    viaAppend.inner().appendRunStartBytes(ra, 0);
    viaAppend.inner().appendBlobStartBytes(ba, 0);
    viaAppend.inner().appendBlobPoolBytes(pa, 0);
    viaAssign.inner().keyStore().appendKeyBytes(kb, 0);
    viaAssign.inner().appendRunStartBytes(rb, 0);
    viaAssign.inner().appendBlobStartBytes(bb, 0);
    viaAssign.inner().appendBlobPoolBytes(pb, 0);
    ASSERT_TRUE(ka == kb);
    ASSERT_TRUE(ra == rb);
    ASSERT_TRUE(ba == bb);
    ASSERT_TRUE(pa == pb);

    // The decoded run for a non-last key is initial + appended, in order.
    const int32_t id3 = viaAppend.lookup(int16_t(3));
    ASSERT_TRUE(id3 != 0);
    ASSERT_EQ(viaAppend.runLen(id3), 3);
    const std::vector<TestRecord> run3 = viaAppend.recordsAt(id3);
    ASSERT_TRUE((run3[0] == TestRecord{ 3, { 3, 6 } }));
    ASSERT_TRUE((run3[2] == TestRecord{ 503, { 9 } }));

    // The brand-new key minted by appendRecord round-trips.
    const int32_t id13 = viaAppend.lookup(int16_t(13));
    ASSERT_TRUE(id13 != 0);
    ASSERT_EQ(viaAppend.runLen(id13), 1);
    ASSERT_TRUE((viaAppend.recordAt(id13, 0) == TestRecord{ 1313, { 13 } }));
}

TEST(typed_cold_map, blob_peek_matches_record_decode) {
    // The zero-allocation peek (peekRecordBytes -- the owner-set prune's read
    // door) must yield bytes that decode to the same record recordAt returns, on
    // BOTH the single-page contiguous path AND the page-straddling fallback.
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);                        // page = 8192 bytes (default)
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int16_t, TestRecord> m(&lb, &d);

    // Small run-length-1 records: every blob fits one page -> contiguous peek.
    for (int16_t k = 1; k <= 20; ++k)
        m.assignRun(k, std::vector<TestRecord>{
            TestRecord{ int32_t(k * 7),
                        { k, int32_t(k + 1), int32_t(k + 2) } } });

    // One record whose blob exceeds a page (8192 B): 2100 int32 ~ 8408 B, so it
    // straddles >= 2 pages and exercises the readBlob/scratch fallback.
    std::vector<int32_t> big(2100);
    for (int i = 0; i < 2100; ++i)
        big[static_cast<std::size_t>(i)] = i * 3 - 5;
    m.assignRun(int16_t(999), std::vector<TestRecord>{ TestRecord{ 4242, big } });

    std::vector<char> scratch;
    const auto checkKey = [&](int16_t k) {
        const int32_t id = m.lookup(k);
        ASSERT_TRUE(id != 0);
        ASSERT_EQ(m.runLen(id), 1);
        int32_t len = 0;
        const char* p = m.peekRecordBytes(id, 0, len, scratch);
        const TestRecord peeked =
            gl::Codec<TestRecord>::deserialize(p, len);
        ASSERT_TRUE(peeked == m.recordAt(id, 0));
    };
    for (int16_t k = 1; k <= 20; ++k) checkKey(k);
    checkKey(int16_t(999));                   // the straddling blob

    // Direct contiguity check: a small blob is contiguous, the big one straddles.
    const int32_t idSmall = m.lookup(int16_t(3));
    const int32_t idBig = m.lookup(int16_t(999));
    const char* pp = nullptr;
    int32_t ll = 0;
    ASSERT_TRUE(m.inner().peekBlobAt(idSmall, 0, pp, ll));   // single page
    ASSERT_TRUE(!m.inner().peekBlobAt(idBig, 0, pp, ll));    // straddles pages
}

TEST(typed_cold_map, blob_peek_arena_matches_vector) {
    // The ScratchArena& peek overload (TypedCold::peekRecordBytes and the
    // underlying HashMap::peekBlobContiguous) must return BYTE-IDENTICAL bytes
    // to the std::vector<char> overload -- the ORACLE -- on BOTH the single-page
    // contiguous path AND a forced page straddle. This is the substrate twin the
    // equi-class transient statification (I-138)
    // rests on: cleanUpExpressions / updateEquivalenceClasses /
    // applyEquivalenceClassToNegatedEquality peek onto a scratch arena instead
    // of an owned std::vector<char>.
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);                        // page = 8192 B, block = 256 KiB
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int16_t, TestRecord> m(&lb, &d);

    // 20 contiguous small records + one whose blob straddles the 8192-B page
    // (2100 int32 ~ 8408 B). 8408 is far under the 256-KiB byte-bump block, so
    // the arena assembles the straddle in ONE alloc (no block-ceiling assert).
    for (int16_t k = 1; k <= 20; ++k)
        m.assignRun(k, std::vector<TestRecord>{
            TestRecord{ int32_t(k * 7),
                        { k, int32_t(k + 1), int32_t(k + 2) } } });
    std::vector<int32_t> big(2100);
    for (int i = 0; i < 2100; ++i)
        big[static_cast<std::size_t>(i)] = i * 3 - 5;
    m.assignRun(int16_t(999), std::vector<TestRecord>{ TestRecord{ 4242, big } });

    // ScratchArena === LbArena; bind a distinct byte-bump arena to the pool.
    gl::ScratchArena scr;
    scr.bind(&g);

    const auto checkKey = [&](int16_t k, bool expectStraddle) {
        const int32_t id = m.lookup(k);
        ASSERT_TRUE(id != 0);
        ASSERT_EQ(m.runLen(id), 1);

        // Oracle: the std::vector<char> overload.
        std::vector<char> vscratch;
        int32_t vlen = 0;
        const char* vp = m.peekRecordBytes(id, 0, vlen, vscratch);

        // Contiguity contract: contiguous <=> !expectStraddle.
        const char* peekp = nullptr;
        int32_t peekl = 0;
        const bool contiguous = m.inner().peekBlobAt(id, 0, peekp, peekl);
        ASSERT_TRUE(contiguous == !expectStraddle);

        // Arena overload at the TypedCold level (updateEquivalenceClasses site).
        int32_t alen = 0;
        const char* ap = m.peekRecordBytes(id, 0, alen, scr);
        ASSERT_EQ(alen, vlen);
        ASSERT_TRUE(std::memcmp(ap, vp, static_cast<std::size_t>(vlen)) == 0);

        // Arena overload at the HashMap level (cleanUpExpressions /
        // applyEquivalenceClassToNegatedEquality use .inner().peekBlobContiguous).
        int32_t hlen = 0;
        const char* hp = m.inner().peekBlobContiguous(id, 0, hlen, scr);
        ASSERT_EQ(hlen, vlen);
        ASSERT_TRUE(std::memcmp(hp, vp, static_cast<std::size_t>(vlen)) == 0);

        if (contiguous) {
            // Zero-copy: both overloads return the in-place pool pointer and
            // never touch the arena.
            ASSERT_TRUE(ap == peekp);
            ASSERT_TRUE(hp == peekp);
        } else {
            // Straddle: assembled into the arena, a fresh buffer distinct from
            // the oracle's vector copy, byte-identical to it.
            ASSERT_TRUE(ap != vp);
        }
    };
    for (int16_t k = 1; k <= 20; ++k) checkKey(k, false);
    checkKey(int16_t(999), true);             // the straddling blob -> arena
}

TEST(typed_cold_map, blob_peek_buffer_matches_vector) {
    // The caller-buffer peek overload (TypedCold::peekRecordBytes and the
    // underlying HashMap::peekBlobContiguous taking char* buf + int32_t cap)
    // must return BYTE-IDENTICAL bytes to the std::vector<char> overload --
    // the ORACLE -- on BOTH the single-page contiguous path AND a forced page
    // straddle. This is the substrate twin the no-arena origin RMWs
    // (addMailOriginRecord / the cold addOriginId) rest on: fixed-capacity
    // stack buffers instead of an owned std::vector<char> or a scratch arena.
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);                        // page = 8192 B, block = 256 KiB
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<int16_t, TestRecord> m(&lb, &d);

    // 20 contiguous small records + one whose blob straddles the 8192-B page
    // (2100 int32 ~ 8408 B), same corpus as the arena twin above.
    for (int16_t k = 1; k <= 20; ++k)
        m.assignRun(k, std::vector<TestRecord>{
            TestRecord{ int32_t(k * 7),
                        { k, int32_t(k + 1), int32_t(k + 2) } } });
    std::vector<int32_t> big(2100);
    for (int i = 0; i < 2100; ++i)
        big[static_cast<std::size_t>(i)] = i * 3 - 5;
    m.assignRun(int16_t(999), std::vector<TestRecord>{ TestRecord{ 4242, big } });

    // A caller buffer big enough for the straddling blob (~8412 B).
    static char buf[16384];

    const auto checkKey = [&](int16_t k, bool expectStraddle) {
        const int32_t id = m.lookup(k);
        ASSERT_TRUE(id != 0);
        ASSERT_EQ(m.runLen(id), 1);

        // Oracle: the std::vector<char> overload.
        std::vector<char> vscratch;
        int32_t vlen = 0;
        const char* vp = m.peekRecordBytes(id, 0, vlen, vscratch);

        // Contiguity contract: contiguous <=> !expectStraddle.
        const char* peekp = nullptr;
        int32_t peekl = 0;
        const bool contiguous = m.inner().peekBlobAt(id, 0, peekp, peekl);
        ASSERT_TRUE(contiguous == !expectStraddle);

        // Caller-buffer overload at the TypedCold level (the origin-RMW site).
        int32_t blen = 0;
        const char* bp = m.peekRecordBytes(id, 0, blen, buf,
                                           int32_t(sizeof(buf)));
        ASSERT_EQ(blen, vlen);
        ASSERT_TRUE(std::memcmp(bp, vp, static_cast<std::size_t>(vlen)) == 0);

        // Caller-buffer overload at the HashMap level.
        int32_t hlen = 0;
        const char* hp = m.inner().peekBlobContiguous(id, 0, hlen, buf,
                                                      int32_t(sizeof(buf)));
        ASSERT_EQ(hlen, vlen);
        ASSERT_TRUE(std::memcmp(hp, vp, static_cast<std::size_t>(vlen)) == 0);

        if (contiguous) {
            // Zero-copy: both return the in-place pool pointer, buf untouched.
            ASSERT_TRUE(bp == peekp);
            ASSERT_TRUE(hp == peekp);
        } else {
            // Straddle: assembled into the caller buffer, byte-identical to
            // the oracle's vector copy.
            ASSERT_TRUE(bp == buf);
            ASSERT_TRUE(hp == buf);
        }
    };
    for (int16_t k = 1; k <= 20; ++k) checkKey(k, false);
    checkKey(int16_t(999), true);             // the straddling blob -> buf
}

// ---- 3. Re-exported facets bind to inner() and reload (the LbMemory wiring) -

TEST(typed_cold_map, facets_reexport_dump_and_reload) {
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lbA(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::TypedColdMap<gl::StatementKey, int> a(&lbA, &dA);
    for (int16_t i = 0; i < 70; ++i)
        a.insert(gl::StatementKey{ int16_t(i + 1), int16_t(i + 3) }, i + 1);

    // Dump through the wrapper's re-exported facet types, bound to inner() —
    // exactly the construction LbMemory uses.
    using TM = gl::TypedColdMap<gl::StatementKey, int>;
    TM::KeysView aKeys(&a.inner());
    TM::ValuesView aVals(&a.inner());
    std::vector<char> kb, vb;
    aKeys.appendSpanBytes(kb, 0);
    aVals.appendSpanBytes(vb, 0);

    // Reload onto a fresh arena through the facets, key tag first.
    gl::LbArena lbB(&g);
    gl::DirtyState dB = gl::DirtyState::Clean;
    TM b(&lbB, &dB);
    TM::KeysView bKeys(&b.inner());
    TM::ValuesView bVals(&b.inner());
    bKeys.bulkAppendBytes(kb.data(), aKeys.size());
    bVals.bulkAppendBytes(vb.data(), aVals.size());

    ASSERT_EQ(b.count(), a.count());
    for (int16_t i = 0; i < 70; ++i) {
        const gl::StatementKey k{ int16_t(i + 1), int16_t(i + 3) };
        const int* p = b.find(k);
        ASSERT_TRUE(p != nullptr && *p == i + 1);
    }
    // Re-dump: the image is a pure function of content.
    std::vector<char> kb2, vb2;
    TM::KeysView(&b.inner()).appendSpanBytes(kb2, 0);
    TM::ValuesView(&b.inner()).appendSpanBytes(vb2, 0);
    ASSERT_TRUE(kb2 == kb);
    ASSERT_TRUE(vb2 == vb);
}

TEST(typed_cold_map, copy_from_deep_copy) {
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdMap<gl::StatementKey, int> src(&lb, &d);
    for (int16_t i = 0; i < 25; ++i)
        src.insert(gl::StatementKey{ int16_t(2 * i + 1), int16_t(i) }, i + 1);

    gl::LbArena lbClone(&g);
    gl::DirtyState dClone = gl::DirtyState::Clean;
    gl::TypedColdMap<gl::StatementKey, int> clone(&lbClone, &dClone);
    clone.copyFrom(src);
    ASSERT_EQ(clone.count(), src.count());
    for (int16_t i = 0; i < 25; ++i) {
        const int* v = clone.find(gl::StatementKey{ int16_t(2 * i + 1),
                                                    int16_t(i) });
        ASSERT_TRUE(v != nullptr && *v == i + 1);
    }
}

TEST(typed_cold_map, set_map_raw_keyview_overloads) {
    // The packed-int set-maps (intToBeProved / intStatementLevelsMap) reach the
    // wrapper through both the typed key and the raw pre-computed packed scalar;
    // the two overload families must resolve to the same key.
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdSetMap<gl::StatementKey, int32_t> m(&lb, &d);

    const gl::StatementKey k{ 5, 9 };
    const int64_t packed = gl::Codec<gl::StatementKey>::encode(k);

    m.insertSorted(packed, 3);          // raw overload
    m.insertSorted(k, 1);               // typed overload — same key
    m.insertSorted(packed, 2);          // raw
    ASSERT_EQ(m.lookup(packed), m.lookup(k));   // resolve to one id
    const int32_t id = m.lookup(k);
    ASSERT_EQ(m.runLen(id), 3);                 // {1,2,3} sorted-unique
    ASSERT_TRUE(m.setContains(packed, 2));
    ASSERT_TRUE(m.setContains(k, 1));

    std::vector<int32_t> vs{ 7, 7, 4 };
    m.assignSetRange(packed, vs.begin(), vs.end());   // raw range replace
    ASSERT_EQ(m.runLen(m.lookup(k)), 2);              // {4,7}
    ASSERT_TRUE(m.eraseSet(packed));                  // raw erase
    ASSERT_EQ(m.lookup(k), 0);
}

// The eradicate-edge inner() raw-door probe (S5 C6): assembling the
// count-prefixed Codec<IdVecKey> bytes by hand and probing / erasing
// through inner() is byte-equivalent to the typed contains / erase.
// Chains include the prefix trap ([1,2] vs [1,2,3] — the count prefix is
// what keeps them distinct byte keys), a negative id, and the EMPTY chain;
// plus a never-minted miss probe on both paths.
TEST(typed_cold_map, idveckey_inner_door_probe_matches_typed) {
    gl::GlobalMemoryManager g;
    g.init(kTypedCfg);
    gl::LbArena lbA(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::TypedColdSet<gl::IdVecKey> setA(&lbA, &dA);   // typed path
    gl::LbArena lbB(&g);
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdSet<gl::IdVecKey> setB(&lbB, &dB);   // inner-door path

    const std::vector<std::vector<int32_t>> chains = {
        { 1, 2 }, { 1, 2, 3 }, { -7 }, {} };
    for (const auto& ids : chains) {
        setA.mint(gl::IdVecKey{ ids });
        setB.mint(gl::IdVecKey{ ids });
    }

    // Hand-assembled count-prefixed bytes — the production probe shape.
    const auto assembled = [](const std::vector<int32_t>& ids) {
        const int32_t cnt = static_cast<int32_t>(ids.size());
        std::string bytes(4 * (ids.size() + 1), '\0');
        std::memcpy(&bytes[0], &cnt, 4);
        for (std::size_t i = 0; i < ids.size(); ++i)
            std::memcpy(&bytes[4 * (i + 1)], &ids[i], 4);
        return bytes;
    };

    // contains parity on every chain + a never-minted miss probe.
    std::vector<std::vector<int32_t>> probes = chains;
    probes.push_back({ 9, 9, 9 });   // miss
    for (const auto& ids : probes) {
        const std::string bytes = assembled(ids);
        const bool typedHit = setA.contains(gl::IdVecKey{ ids });
        const bool doorHit = setB.inner().contains(
            gl::StrSpan(bytes.data(), static_cast<int32_t>(bytes.size())));
        ASSERT_EQ(typedHit, doorHit);
    }

    // Erase one chain via the door on B vs typed on A; then a miss erase on
    // both paths; facet equality afterwards (count + per-id decoded bytes).
    {
        const std::string bytes = assembled({ 1, 2 });
        ASSERT_TRUE(setA.erase(gl::IdVecKey{ { 1, 2 } }));
        ASSERT_TRUE(setB.inner().erase(
            gl::StrSpan(bytes.data(), static_cast<int32_t>(bytes.size()))));
        const std::string missBytes = assembled({ 9, 9, 9 });
        ASSERT_FALSE(setA.erase(gl::IdVecKey{ { 9, 9, 9 } }));
        ASSERT_FALSE(setB.inner().erase(
            gl::StrSpan(missBytes.data(),
                        static_cast<int32_t>(missBytes.size()))));
    }
    ASSERT_EQ(setA.count(), setB.count());
    for (int32_t id = 1; id <= setA.count(); ++id) {
        const gl::StrSpan ka = setA.inner().keyAt(id);
        const gl::StrSpan kb = setB.inner().keyAt(id);
        ASSERT_EQ(ka.len, kb.len);
        ASSERT_EQ(std::memcmp(ka.ptr, kb.ptr,
                              static_cast<std::size_t>(ka.len)), 0);
    }
    // The prefix-trap sibling [1,2,3] survived both erases.
    ASSERT_TRUE(setA.contains(gl::IdVecKey{ { 1, 2, 3 } }));
    ASSERT_TRUE(setB.contains(gl::IdVecKey{ { 1, 2, 3 } }));
}
