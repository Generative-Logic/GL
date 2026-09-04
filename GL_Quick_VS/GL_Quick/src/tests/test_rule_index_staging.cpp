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
/// @brief Differential tests for the rule-index staging
///        (D-333): every staging twin plus its flush
///        reproduces the sequential cold doors byte for byte.
///
/// @details
/// 1. `BlobCsrValueStore::rebuildRunsGenerated` (through
///    `HashMap::assignAllRunsGenerated`) rebuilds the value columns of a map
///    to the bytes a sequence of `assignRun` / `appendBlobToRun` splices left.
/// 2. Owner runs: random (key, owner) sequences through `addOwnerToRun`
///    versus `stageOwnerToRun` + `flushOwnerSection`, in two rounds so the
///    second flush merges into existing runs.
/// 3. LMV runs: `appendLmvIdsRecord` versus `stageLmvIdsRecord` +
///    `flushLmvSection` (arrival order kept).
/// 4. Subkey records: `addShortSubkeyOwner` / `mergeSubkeySignatures` versus
///    the staging twins + `flushSubkeySection` (loose byte, sorted-unique
///    signatures, renumbered sorted-unique pairs).
/// 5. Remaining-args runs: `insertRemainingArgsNormKeyBatch` versus
///    `stageRemainingArgsNormKeyBatch` + `flushRemArgsSection` /
///    `flushOwnerSection` — forward runs, edge owners and reverse-index
///    answers.
/// 6. An empty staging flushes as a no-op (nothing dirtied, nothing minted).
///
/// Every corpus comes from a fixed-seed linear congruential sequence, so the
/// tests are byte-reproducible across hosts and runs.

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

    // Deterministic 32-bit source (high bits — an LCG's low bits cycle fast).
    uint32_t nextRandom(uint32_t& s) {
        s = s * 1664525u + 1013904223u;
        return s >> 16;
    }

    // Byte image of a blob map: the key count, every key's bytes in id order,
    // every run's blobs. Two maps with equal images are byte-identical.
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

    // A NormKey's cold key bytes for a small synthetic payload.
    int32_t normKeyBytes(int32_t numberExpressions, const gl::NameId* data, int32_t len,
                         char* buf) {
        return gl::encodeNormKeyInto(numberExpressions, data, len, buf,
                                     gl::ExecutionParameters::kMaxNormKeyBytes);
    }

}  // namespace

// 1. One rebuild reproduces the columns a random splice sequence left; a
//    rebuild that reads the map's own runs is a fixed point; a key minted
//    after the last write receives its run.
TEST(rule_index_staging, rebuild_all_runs_matches_splice_sequence) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena scratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::HashMap<gl::BytesKeyStore, gl::BlobCsrValueStore> a(&lbA, &dA);
    gl::HashMap<gl::BytesKeyStore, gl::BlobCsrValueStore> b(&lbB, &dB);

    uint32_t seed = 20260903u;
    const char* keys[5] = { "alpha", "b", "gamma-key", "dd", "e5" };
    for (int t = 0; t < 300; ++t) {
        const int ki = static_cast<int>(nextRandom(seed) % 5u);
        const gl::StrSpan k(keys[ki], static_cast<int32_t>(std::strlen(keys[ki])));
        char blob[16];
        const int32_t len = 1 + static_cast<int32_t>(nextRandom(seed) % 15u);
        for (int32_t i = 0; i < len; ++i)
            blob[i] = static_cast<char>('a' + (nextRandom(seed) % 26u));
        const int32_t id = a.lookup(k);
        if (id != 0) a.appendBlobToRun(id, blob, len);
        else a.assignRun(k, blob, &len, 1);
    }
    ASSERT_EQ(a.count(), 5);

    // The runs first (for the five keys about to be minted), then the keys in
    // A's id order — the flush's own order.
    b.assignAllRunsGenerated(a.count(), [&](int32_t id, auto&& sink) {
        a.forEachBlobContiguous(id, scratch,
            [&](const char* p, int32_t l) { sink.blob(p, l); });
    }, scratch);
    for (int32_t id = 1; id <= a.count(); ++id) b.mint(a.keyAt(id));
    ASSERT_TRUE(imageOf(a) == imageOf(b));
    ASSERT_TRUE(dB == gl::DirtyState::Restructured);

    // Fixed point over its own runs through the wholesale copy (the range
    // 1..3, then key 5) and the record path (key 4), plus a late key with a
    // fresh run.
    const std::vector<char> before = imageOf(b);
    b.assignAllRunsGenerated(a.count() + 1, [&](int32_t id, auto&& sink) {
        if (id > a.count()) {
            sink.blob("zz", 2);
        } else if (id != 4) {
            sink.unchanged(id);
        } else {
            b.forEachBlobContiguous(id, scratch,
                [&](const char* p, int32_t l) { sink.blob(p, l); });
        }
    }, scratch);
    b.mint(gl::StrSpan("late", 4));
    ASSERT_EQ(b.count(), 6);
    {
        const std::vector<char> after = imageOf(b);
        ASSERT_TRUE(after.size() > before.size());
        ASSERT_TRUE(std::equal(before.begin() + 4, before.end(), after.begin() + 4));
    }
    ASSERT_EQ(b.runLen(6), 1);
    std::vector<char> late;
    b.blobAt(6, 0, late);
    ASSERT_EQ(static_cast<int>(late.size()), 2);
    ASSERT_TRUE(late[0] == 'z' && late[1] == 'z');
    for (int32_t id = 1; id <= 5; ++id) {
        ASSERT_EQ(a.runLen(id), b.runLen(id));
        for (int32_t j = 0; j < a.runLen(id); ++j) {
            std::vector<char> x, y;
            a.blobAt(id, j, x);
            b.blobAt(id, j, y);
            ASSERT_TRUE(x == y);
        }
    }
    (void)before;
}

// 2. Owner runs: two rounds of random (key, owner) pairs; the staged flush
//    equals the sequential doors after each round (merge into existing runs).
TEST(rule_index_staging, owner_section_flush_matches_sequential_doors) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena scratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::RuleOwnerRec> mapA(&lbA, &dA);
    gl::TypedColdBlobMap<gl::NormKey, gl::RuleOwnerRec> mapB(&lbB, &dB);
    gl::RuleIndexStagingSection section(&lbB);

    uint32_t seed = 20260904u;
    char keyBufs[6][gl::ExecutionParameters::kMaxNormKeyBytes];
    int32_t keyLens[6];
    for (int32_t k = 0; k < 6; ++k) {
        gl::NameId data[4] = { static_cast<gl::NameId>(10 + k), 0,
                               static_cast<gl::NameId>(1 + (k % 3)), 0 };
        keyLens[k] = normKeyBytes(1 + (k % 2), data, 2 + 2 * (k % 2), keyBufs[k]);
    }
    for (int round = 0; round < 2; ++round) {
        for (int t = 0; t < 150; ++t) {
            const int32_t k = static_cast<int32_t>(nextRandom(seed) % 6u);
            const gl::RuleOwner owner = gl::packRuleOwner(
                static_cast<int32_t>(1 + nextRandom(seed) % 12u),
                static_cast<gl::NameId>(1 + nextRandom(seed) % 3u));
            const gl::StrSpan key(keyBufs[k], keyLens[k]);
            gl::ExpressionAnalyzer::addOwnerToRun(mapA, key, owner, scratch);
            gl::ExpressionAnalyzer::stageOwnerToRun(section, key, owner);
        }
        ASSERT_FALSE(section.empty());
        gl::ExpressionAnalyzer::flushOwnerSection(mapB, section, scratch);
        section.clear();
        ASSERT_TRUE(section.empty());
        ASSERT_TRUE(imageOf(mapA.inner()) == imageOf(mapB.inner()));
    }
    ASSERT_EQ(mapB.count(), 6);
}

// 3. LMV runs: the staged records land in staging order behind the existing
//    run, byte-identical to the sequential appends.
TEST(rule_index_staging, lmv_section_flush_keeps_arrival_order) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena scratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> mapA(&lbA, &dA);
    gl::TypedColdBlobMap<gl::NormKey, gl::LocalMemoryValue> mapB(&lbB, &dB);
    gl::RuleIndexStagingSection section(&lbB);

    uint32_t seed = 20260905u;
    for (int round = 0; round < 2; ++round) {
        for (int t = 0; t < 120; ++t) {
            const int32_t k = static_cast<int32_t>(nextRandom(seed) % 5u);
            gl::NameId data[2] = { static_cast<gl::NameId>(20 + k), 0 };
            const int32_t valueId = static_cast<int32_t>(1 + nextRandom(seed) % 50u);
            const bool isMarker = (nextRandom(seed) % 2u) == 1u;
            int32_t keyIds[3];
            const int32_t keyN = static_cast<int32_t>(nextRandom(seed) % 4u);
            for (int32_t i = 0; i < keyN; ++i) keyIds[i] = static_cast<int32_t>(nextRandom(seed) % 90u);
            int32_t remIds[2];
            const int32_t remN = static_cast<int32_t>(nextRandom(seed) % 3u);
            for (int32_t i = 0; i < remN; ++i) remIds[i] = static_cast<int32_t>(nextRandom(seed) % 90u);
            int levels[2];
            const int32_t levelN = static_cast<int32_t>(nextRandom(seed) % 3u);
            for (int32_t i = 0; i < levelN; ++i) levels[i] = 2 * i + static_cast<int>(nextRandom(seed) % 2u);
            const int32_t origId = static_cast<int32_t>(1 + nextRandom(seed) % 30u);
            const gl::NameId vid = static_cast<gl::NameId>(1 + nextRandom(seed) % 4u);
            const gl::RuleJustification just = (nextRandom(seed) % 2u) == 0u
                ? gl::RuleJustification::none : gl::RuleJustification::integration;
            const bool productOf = (nextRandom(seed) % 2u) == 1u;
            const bool disint = (nextRandom(seed) % 2u) == 1u;
            const bool ordisOnly = (nextRandom(seed) % 2u) == 1u;
            const bool subsetExcl = (nextRandom(seed) % 2u) == 1u;
            const bool ordis2 = (nextRandom(seed) % 2u) == 1u;
            gl::appendLmvIdsRecord(mapA, 1, data, 2, valueId, isMarker, keyIds, keyN,
                remIds, remN, origId, vid, scratch, levels, levelN, just, productOf,
                disint, ordisOnly, subsetExcl, ordis2);
            gl::stageLmvIdsRecord(section, 1, data, 2, valueId, isMarker, keyIds, keyN,
                remIds, remN, origId, vid, scratch, levels, levelN, just, productOf,
                disint, ordisOnly, subsetExcl, ordis2);
        }
        gl::ExpressionAnalyzer::flushLmvSection(mapB, section, scratch);
        section.clear();
        ASSERT_TRUE(imageOf(mapA.inner()) == imageOf(mapB.inner()));
    }
    ASSERT_EQ(mapB.count(), 5);
    ASSERT_TRUE(mapB.inner().blobCount() == 240);
}

// 4. Subkey records: short (loose) and long (signature) contributions on
//    distinct keys, two rounds; the merged record equals the sequential one.
TEST(rule_index_staging, subkey_section_flush_matches_sequential_merges) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena scratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> mapA(&lbA, &dA);
    gl::TypedColdBlobMap<gl::NormKey, gl::OwnerSet> mapB(&lbB, &dB);
    gl::RuleIndexStagingSection section(&lbB);

    uint32_t seed = 20260906u;
    gl::IntEncodedExpr pool[8];
    for (int p = 0; p < 8; ++p) {
        gl::IntEncodedExpr e{};
        e.nameId = static_cast<gl::NameId>(10 + p);
        e.arity = 1 + static_cast<int32_t>(nextRandom(seed) % 3u);
        for (int32_t j = 0; j < e.arity; ++j) {
            e.argId[j] = static_cast<gl::NameId>(100 + nextRandom(seed) % 4u);
            e.argUnchangeable[j] = (nextRandom(seed) % 3u) == 0u ? 1 : 0;
            e.argFullId[j] = static_cast<gl::NameId>(200 + nextRandom(seed) % 5u);
        }
        pool[p] = e;
    }
    // Keys 0..2 are short (one or two premises), keys 3..5 long (three or four):
    // a key's premise count is part of its identity, so the two doors never
    // meet on one key.
    const int32_t countOfKey[6] = { 1, 2, 2, 3, 4, 3 };
    char keyBufs[6][gl::ExecutionParameters::kMaxNormKeyBytes];
    int32_t keyLens[6];
    for (int32_t k = 0; k < 6; ++k) {
        gl::NameId data[2] = { static_cast<gl::NameId>(30 + k), 0 };
        keyLens[k] = normKeyBytes(countOfKey[k], data, 2, keyBufs[k]);
    }
    for (int round = 0; round < 2; ++round) {
        for (int t = 0; t < 200; ++t) {
            const int32_t k = static_cast<int32_t>(nextRandom(seed) % 6u);
            const gl::NameId count = static_cast<gl::NameId>(countOfKey[k]);
            gl::IntEncodedExpr exprs[4];
            for (gl::NameId i = 0; i < count; ++i) exprs[i] = pool[nextRandom(seed) % 8u];
            const gl::RuleOwner owner = gl::packRuleOwner(
                static_cast<int32_t>(1 + nextRandom(seed) % 10u), 1);
            gl::NameId data[2] = { static_cast<gl::NameId>(30 + k), 0 };
            if (count < gl::ExpressionAnalyzer::kSubkeyUCheckMinElements) {
                gl::ExpressionAnalyzer::addShortSubkeyOwner(mapA, count, data, 2, owner);
                gl::ExpressionAnalyzer::stageShortSubkeyOwner(section, count, data, 2, owner);
            } else {
                gl::ExpressionAnalyzer::mergeSubkeySignatures(mapA, count, data, 2,
                                                              exprs, count, owner);
                gl::ExpressionAnalyzer::stageSubkeySignatures(section, count, data, 2,
                                                              exprs, count, owner);
            }
        }
        gl::ExpressionAnalyzer::flushSubkeySection(mapB, section, scratch);
        section.clear();
        ASSERT_TRUE(imageOf(mapA.inner()) == imageOf(mapB.inner()));
    }
    ASSERT_EQ(mapB.count(), 6);
    for (int32_t id = 1; id <= 6; ++id) ASSERT_EQ(mapB.runLen(id), 1);
    (void)keyLens;
}

// 5. Remaining-args runs: per-copy batches through the door versus the
//    staged twin; forward runs, edge owners and reverse-index answers agree.
TEST(rule_index_staging, remargs_section_flush_matches_batch_door) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lbA(&g);
    gl::LbArena lbB(&g);
    gl::LbArena scratch(&g);
    gl::DirtyState dA = gl::DirtyState::Clean;
    gl::DirtyState dB = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::Int16SetKey, gl::NormKey> mapA(&lbA, &dA);
    gl::TypedColdBlobMap<gl::Int16SetKey, gl::NormKey> mapB(&lbB, &dB);
    gl::ReverseArgsIndex revA(&lbA);
    gl::ReverseArgsIndex revB(&lbB);
    gl::TypedColdBlobMap<gl::IdVecKey, gl::RuleOwnerRec> ownersA(&lbA, &dA);
    gl::TypedColdBlobMap<gl::IdVecKey, gl::RuleOwnerRec> ownersB(&lbB, &dB);
    gl::RuleIndexStagingSection secRemArgs(&lbB);
    gl::RuleIndexStagingSection secOwners(&lbB);

    uint32_t seed = 20260907u;
    std::vector<gl::NormKey> poolKeys;
    for (int p = 0; p < 12; ++p) {
        gl::NormKey nk;
        nk.numberExpressions = 1 + static_cast<int32_t>(nextRandom(seed) % 3u);
        const int32_t n = 1 + static_cast<int32_t>(nextRandom(seed) % 3u);
        for (int32_t i = 0; i < n; ++i)
            nk.data.push_back(static_cast<gl::NameId>(5 + nextRandom(seed) % 9u));
        poolKeys.push_back(nk);
    }
    const std::vector<std::vector<gl::NameId>> argSets = { { 1, 2 }, { 2, 5, 7 }, { 9 } };

    const auto flushB = [&]() {
        gl::ExpressionAnalyzer::flushRemArgsSection(mapB, revB, secRemArgs, scratch);
        gl::ExpressionAnalyzer::flushOwnerSection(ownersB, secOwners, scratch);
        secRemArgs.clear();
        secOwners.clear();
    };
    for (int copy = 0; copy < 60; ++copy) {
        const std::vector<gl::NameId>& argSet = argSets[nextRandom(seed) % 3u];
        gl::NameId argArr[3];
        for (std::size_t i = 0; i < argSet.size(); ++i) argArr[i] = argSet[i];
        const int32_t argN = static_cast<int32_t>(argSet.size());
        const gl::RuleOwner owner = gl::packRuleOwner(
            static_cast<int32_t>(1 + nextRandom(seed) % 5u), 1);
        const int32_t batchN = 1 + static_cast<int32_t>(nextRandom(seed) % 6u);
        const gl::ArenaOffset bMark = scratch.cursor();
        {
            gl::DirtyState bDirty = gl::DirtyState::Clean;
            gl::PagedVector<gl::ExpressionAnalyzer::RemArgsBatchBlob> batch(&scratch, &bDirty);
            for (int32_t i = 0; i < batchN; ++i) {
                const std::vector<char> bytes =
                    gl::Codec<gl::NormKey>::serialize(poolKeys[nextRandom(seed) % 12u]);
                const int32_t bl = static_cast<int32_t>(bytes.size());
                const gl::ArenaOffset off = scratch.alloc(bl, 1);
                std::memcpy(scratch.resolve(off), bytes.data(), static_cast<std::size_t>(bl));
                batch.push_back(gl::ExpressionAnalyzer::RemArgsBatchBlob{ off, bl });
            }
            gl::ExpressionAnalyzer::insertRemainingArgsNormKeyBatch(
                mapA, revA, ownersA, argArr, argN, batch, owner, scratch);
            gl::ExpressionAnalyzer::stageRemainingArgsNormKeyBatch(
                secRemArgs, secOwners, argArr, argN, batch, owner, scratch);
        }
        scratch.popTo(bMark);
        if (copy == 29) flushB();
    }
    flushB();

    ASSERT_TRUE(imageOf(mapA.inner()) == imageOf(mapB.inner()));
    ASSERT_TRUE(imageOf(ownersA.inner()) == imageOf(ownersB.inner()));
    ASSERT_EQ(mapB.count(), 3);
    for (const gl::NormKey& nk : poolKeys) {
        const std::vector<char> b = gl::Codec<gl::NormKey>::serialize(nk);
        std::vector<int32_t> oa, ob;
        revA.reverseIndexRunOf(gl::StrSpan(b.data(), static_cast<int32_t>(b.size())),
            [&](int32_t id) { oa.push_back(id); });
        revB.reverseIndexRunOf(gl::StrSpan(b.data(), static_cast<int32_t>(b.size())),
            [&](int32_t id) { ob.push_back(id); });
        std::sort(oa.begin(), oa.end());
        std::sort(ob.begin(), ob.end());
        ASSERT_TRUE(oa == ob);
    }
}

// 6. An empty staging flushes as a no-op: nothing minted, nothing dirtied.
TEST(rule_index_staging, empty_flush_is_a_no_op) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 16 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::HashMemory hm(&lb, &d);
    gl::LbArena stagingArena(&g);
    gl::RuleIndexStaging st(&stagingArena);
    ASSERT_TRUE(st.empty());
    gl::ExpressionAnalyzer::flushRuleIndexStaging(hm, st);
    ASSERT_TRUE(d == gl::DirtyState::Clean);
    ASSERT_EQ(hm.encodedMap.count(), 0);
    ASSERT_EQ(hm.normalizedEncodedKeys.count(), 0);
    ASSERT_EQ(hm.normalizedEncodedSubkeys.count(), 0);
    ASSERT_EQ(hm.remainingArgsNormalizedEncodedMap.count(), 0);

    // One staged owner: the flush mints the key, writes the run, empties the
    // staging and keeps its binding (a window close releases it).
    gl::NameId data[2] = { 7, 0 };
    st.boundTo = &hm;
    gl::ExpressionAnalyzer::stageWholeKeyOwner(st.wholeKeys, 1, data, 2,
                                               gl::packRuleOwner(3, 1));
    ASSERT_FALSE(st.empty());
    gl::ExpressionAnalyzer::flushRuleIndexStaging(hm, st);
    ASSERT_TRUE(st.empty());
    ASSERT_TRUE(st.boundTo == &hm);   // a flush keeps the binding; only a window close releases it
    ASSERT_EQ(hm.normalizedEncodedKeys.count(), 1);
    ASSERT_EQ(hm.normalizedEncodedKeys.runLen(1), 1);
    ASSERT_EQ(hm.normalizedEncodedKeys.recordAt(1, 0).owner, gl::packRuleOwner(3, 1));
    hm.releaseAllCold();
}

// 7. Beyond one block: a map with more keys than one block of int32 slots
//    (the plan table) and one key with more staged items than one sort chunk
//    (the block-bounded sort) — both flush exactly, checked against a plain
//    sorted-unique oracle.
TEST(rule_index_staging, flush_scales_past_one_block) {
    gl::GlobalMemoryManager g;
    g.init(gl::StaticMemoryConfig{ 64 << 20, 1 << 18 });
    gl::LbArena lb(&g);
    gl::LbArena scratch(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::TypedColdBlobMap<gl::NormKey, gl::RuleOwnerRec> map(&lb, &d);
    gl::RuleIndexStagingSection section(&lb);

    // 70,000 distinct keys, one owner each: more plan-table entries than a
    // 256 KiB block holds.
    constexpr int32_t kKeys = 70000;
    for (int32_t k = 0; k < kKeys; ++k) {
        gl::NameId data[2] = { static_cast<gl::NameId>(k & 0xFFFF),
                               static_cast<gl::NameId>(k >> 16) };
        gl::ExpressionAnalyzer::stageWholeKeyOwner(section, 2, data, 2,
                                                   gl::packRuleOwner(k + 1, 1));
    }
    gl::ExpressionAnalyzer::flushOwnerSection(map, section, scratch);
    section.clear();
    ASSERT_EQ(map.count(), kKeys);
    for (int32_t k = 0; k < kKeys; k += 9973) {
        gl::NameId data[2] = { static_cast<gl::NameId>(k & 0xFFFF),
                               static_cast<gl::NameId>(k >> 16) };
        const int32_t id = map.lookup(gl::NormKey{ 2, { data[0], data[1] } });
        ASSERT_EQ(id, k + 1);
        ASSERT_EQ(map.runLen(id), 1);
        ASSERT_EQ(map.recordAt(id, 0).owner, gl::packRuleOwner(k + 1, 1));
    }

    // 140,000 staged owners under ONE existing key (id 1), drawn from 50,000
    // values: more ordinals than one sort chunk, with duplicates. Oracle: the
    // sorted-unique union with the key's existing owner.
    uint32_t seed = 20260908u;
    std::set<gl::RuleOwner> oracle{ gl::packRuleOwner(1, 1) };
    gl::NameId first[2] = { 0, 0 };
    for (int32_t i = 0; i < 140000; ++i) {
        const gl::RuleOwner o = gl::packRuleOwner(
            static_cast<int32_t>(2 + nextRandom(seed) % 50000u), 1);
        oracle.insert(o);
        gl::ExpressionAnalyzer::stageWholeKeyOwner(section, 2, first, 2, o);
    }
    gl::ExpressionAnalyzer::flushOwnerSection(map, section, scratch);
    section.clear();
    ASSERT_EQ(map.count(), kKeys);
    ASSERT_EQ(map.runLen(1), static_cast<int32_t>(oracle.size()));
    int32_t j = 0;
    for (const gl::RuleOwner o : oracle) {
        ASSERT_EQ(map.recordAt(1, j).owner, o);
        ++j;
    }
    ASSERT_EQ(map.runLen(2), 1);
    ASSERT_EQ(map.recordAt(2, 0).owner, gl::packRuleOwner(2, 1));
}
