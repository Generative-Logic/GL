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
/// @brief Unit tests for `ReverseArgsIndex` — the derived NormKey ->
///        owning-forward-key reverse membership side-index for
///        `remainingArgsNormalizedEncodedMap`.
///
/// @details
/// The oracle is a brute-force forward scan (`recordsAt` per key), which the
/// production candidate loop is replacing. Each test asserts the reverse index
/// answers the run-contains membership question EXACTLY (no false positives, no
/// omissions) against that oracle. Covered: full rebuild vs brute force,
/// incremental `appendEdge` vs brute force, clear + rebuild identity (the
/// canonical-reload path), and the probe-miss negative case. The test oracle
/// uses `std::vector` freely — it is scaffolding outside the pipeline / CodeQL
/// tree; the container under test is 100% arena-static.

#include "test_harness.hpp"

#include "../memory_infra/reverse_args_index.hpp"
#include "../memory_infra/typed_cold_map.hpp"

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block — the same substrate the sibling tests use.
    const gl::StaticMemoryConfig kRevCfg{ 1 << 20, 1 << 18 };

    using Forward = gl::TypedColdBlobMap<gl::Int16SetKey, gl::NormKey>;

    inline gl::NormKey nk(gl::NameId ne, const std::vector<gl::NameId>& data) {
        return gl::NormKey{ ne, data };
    }
    inline gl::Int16SetKey ik(const std::vector<gl::NameId>& ids) {
        return gl::Int16SetKey{ ids };
    }

    // The NormKey's serialized bytes as an owned string — Codec<NormKey>
    // serialize == key encode; the reverse index probes exactly these bytes.
    inline std::string nkBytes(const gl::NormKey& k) {
        const std::vector<char> v = gl::Codec<gl::NormKey>::serialize(k);
        return std::string(v.begin(), v.end());
    }

    // Brute-force oracle: the forward key ids whose stored run contains `key`,
    // ascending. This is exactly what the production candidate loop's
    // per-candidate byte-peek membership computed before the reverse index.
    inline std::vector<int32_t> bruteForceOwners(const Forward& fwd,
                                                 const gl::NormKey& key) {
        std::vector<int32_t> out;
        const int32_t n = fwd.count();
        for (int32_t id = 1; id <= n; ++id) {
            const std::vector<gl::NormKey> recs = fwd.recordsAt(id);
            if (std::find(recs.begin(), recs.end(), key) != recs.end())
                out.push_back(id);
        }
        return out;
    }

    inline std::vector<int32_t> reverseOwners(const gl::ReverseArgsIndex& rev,
                                              const gl::NormKey& key) {
        const std::string b = nkBytes(key);
        std::vector<int32_t> out;
        rev.reverseIndexRunOf(gl::StrSpan(b), [&](int32_t id) {
            out.push_back(id);
        });
        std::sort(out.begin(), out.end());
        return out;
    }

    // Populate a forward map with `keys[i] -> runs[i]` (each run sorted-unique,
    // the forward map's own invariant), returning the built map's key ids so a
    // caller can drive appendEdge in id order.
    inline void buildForward(Forward& fwd,
                             const std::vector<gl::Int16SetKey>& keys,
                             const std::vector<std::vector<gl::NormKey>>& runs) {
        assert(keys.size() == runs.size());
        for (std::size_t i = 0; i < keys.size(); ++i)
            fwd.assignRun(keys[i], runs[i]);
    }

    // The universe of NormKeys referenced by any run, plus a couple that are
    // never stored (probe-miss coverage).
    inline std::vector<gl::NormKey> probeUniverse() {
        return {
            nk(1, { 10, 20 }),
            nk(1, { 10, 30 }),
            nk(2, { 40 }),
            nk(2, { 40, 50, 60 }),
            nk(3, {}),                 // empty-payload NormKey — a valid key
            nk(9, { 99, 98, 97 }),     // never stored (miss)
            nk(1, { 10, 21 }),         // never stored (hash-neighbourhood miss)
        };
    }
}

// ---- 1. Full rebuild membership == brute-force forward scan ----------------

TEST(reverse_args_index, rebuild_membership_matches_brute_force) {
    gl::GlobalMemoryManager g;
    g.init(kRevCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    Forward fwd(&lb, &d);

    // Deliberate cross-ownership: nk(1,{10,20}) lives under THREE arg-sets,
    // nk(2,{40}) under two, so a correct inversion must collect multiple ids.
    const std::vector<gl::Int16SetKey> keys = {
        ik({ 1, 2 }), ik({ 3 }), ik({ 1, 4, 5 }), ik({ 7 }),
    };
    const std::vector<std::vector<gl::NormKey>> runs = {
        { nk(1, { 10, 20 }), nk(2, { 40 }) },
        { nk(1, { 10, 20 }), nk(1, { 10, 30 }), nk(3, {}) },
        { nk(1, { 10, 20 }), nk(2, { 40, 50, 60 }) },
        { nk(2, { 40 }), nk(3, {}) },
    };
    buildForward(fwd, keys, runs);

    gl::ScratchArena scr;
    scr.bind(&g);
    gl::ReverseArgsIndex rev(&lb);
    const gl::ArenaOffset mark = scr.cursor();
    rev.rebuildReverseIndex(fwd, scr);
    scr.popTo(mark);

    for (const gl::NormKey& key : probeUniverse())
        ASSERT_TRUE(reverseOwners(rev, key) == bruteForceOwners(fwd, key));

    // Positive spot check + negative (never-stored) spot check.
    ASSERT_FALSE(reverseOwners(rev, nk(1, { 10, 20 })).empty());
    ASSERT_TRUE(reverseOwners(rev, nk(9, { 99, 98, 97 })).empty());
    ASSERT_FALSE(rev.empty());
}

// ---- 2. Incremental appendEdge == brute-force forward scan -----------------

TEST(reverse_args_index, append_edge_matches_brute_force) {
    gl::GlobalMemoryManager g;
    g.init(kRevCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    Forward fwd(&lb, &d);

    const std::vector<gl::Int16SetKey> keys = {
        ik({ 2 }), ik({ 5, 6 }), ik({ 1 }),
    };
    const std::vector<std::vector<gl::NormKey>> runs = {
        { nk(1, { 10, 20 }), nk(2, { 40 }) },
        { nk(1, { 10, 20 }), nk(3, {}) },
        { nk(2, { 40 }) },
    };
    buildForward(fwd, keys, runs);

    // Drive the reverse index by the SAME per-record edges
    // insertRemainingArgsNormKey would emit — one appendEdge per (run member,
    // owning key id), never through rebuild.
    gl::ReverseArgsIndex rev(&lb);
    for (int32_t id = 1; id <= fwd.count(); ++id) {
        const std::vector<gl::NormKey> recs = fwd.recordsAt(id);
        for (const gl::NormKey& r : recs) {
            const std::string b = nkBytes(r);
            rev.appendEdge(gl::StrSpan(b), id);
        }
    }

    for (const gl::NormKey& key : probeUniverse())
        ASSERT_TRUE(reverseOwners(rev, key) == bruteForceOwners(fwd, key));
    ASSERT_EQ(rev.distinctKeyCount(), 3);   // {10,20}/1 , {40}/2 , {}/3
}

// ---- 3. clear + rebuild answers identically (the canonical-reload path) ----

TEST(reverse_args_index, clear_then_rebuild_is_identical) {
    gl::GlobalMemoryManager g;
    g.init(kRevCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    Forward fwd(&lb, &d);

    const std::vector<gl::Int16SetKey> keys = {
        ik({ 1, 2 }), ik({ 3 }), ik({ 8, 9 }),
    };
    const std::vector<std::vector<gl::NormKey>> runs = {
        { nk(1, { 10, 20 }), nk(2, { 40 }), nk(3, {}) },
        { nk(1, { 10, 20 }), nk(1, { 10, 30 }) },
        { nk(2, { 40 }), nk(5, { 70, 80 }) },
    };
    buildForward(fwd, keys, runs);

    gl::ScratchArena scr;
    scr.bind(&g);
    gl::ReverseArgsIndex rev(&lb);
    rev.rebuildReverseIndex(fwd, scr);

    // Snapshot every answer, then clear (frees pages, the canonical-release
    // seam) and rebuild (the canonical-reload seam) — answers must be identical.
    std::vector<std::vector<int32_t>> before;
    for (const gl::NormKey& key : probeUniverse())
        before.push_back(reverseOwners(rev, key));

    rev.clear();
    ASSERT_TRUE(rev.empty());
    // A cleared index answers every probe with the empty run (defined miss).
    for (const gl::NormKey& key : probeUniverse())
        ASSERT_TRUE(reverseOwners(rev, key).empty());

    rev.rebuildReverseIndex(fwd, scr);
    for (std::size_t i = 0; i < probeUniverse().size(); ++i)
        ASSERT_TRUE(reverseOwners(rev, probeUniverse()[i]) == before[i]);
}

// removeEdge: unlinking (NormKey -> forward key id) edges leaves the reverse
// answer equal to the brute-force scan of a forward map from which the same
// NormKeys were removed; untouched edges survive, chains with an unlinked
// head, middle and tail node all stay consistent, and a NormKey whose every
// edge left answers empty (its entry stays interned).
TEST(reverse_args_index, remove_edge_matches_brute_force) {
    gl::GlobalMemoryManager g;
    g.init(kRevCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    Forward fwd(&lb, &d);
    gl::ReverseArgsIndex rev(&lb);

    // Three forward keys sharing NormKeys: A in all three, B in two, C in one.
    const gl::NormKey A = nk(1, { 10, 20 });
    const gl::NormKey B = nk(1, { 10, 30 });
    const gl::NormKey C = nk(2, { 40 });
    buildForward(fwd,
        { ik({ 1 }), ik({ 2 }), ik({ 3 }) },
        { { A, B, C }, { A, B }, { A } });
    rev.rebuildReverseIndex(fwd, lb);
    ASSERT_TRUE(reverseOwners(rev, A) == bruteForceOwners(fwd, A));

    // Unlink A from key 2 (a middle node of A's chain -- appends prepend, so
    // the chain is 3,2,1), then C from key 1 (the only node), then A from
    // key 3 (the head) and from key 1 (the tail). Mirror each in the forward map.
    const auto dropFromForward = [&](const gl::Int16SetKey& key, const gl::NormKey& k) {
        const int32_t id = fwd.lookup(key);
        ASSERT_TRUE(id != 0);
        std::vector<gl::NormKey> run = fwd.recordsAt(id);
        run.erase(std::find(run.begin(), run.end(), k));
        fwd.assignRun(key, run);
    };
    dropFromForward(ik({ 2 }), A);
    rev.removeEdge(gl::StrSpan(nkBytes(A)), 2);
    ASSERT_TRUE(reverseOwners(rev, A) == bruteForceOwners(fwd, A));
    ASSERT_TRUE((reverseOwners(rev, A) == std::vector<int32_t>{ 1, 3 }));

    dropFromForward(ik({ 1 }), C);
    rev.removeEdge(gl::StrSpan(nkBytes(C)), 1);
    ASSERT_TRUE(reverseOwners(rev, C).empty());
    ASSERT_TRUE(bruteForceOwners(fwd, C).empty());

    dropFromForward(ik({ 3 }), A);
    rev.removeEdge(gl::StrSpan(nkBytes(A)), 3);
    dropFromForward(ik({ 1 }), A);
    rev.removeEdge(gl::StrSpan(nkBytes(A)), 1);
    ASSERT_TRUE(reverseOwners(rev, A).empty());

    // B untouched throughout.
    ASSERT_TRUE((reverseOwners(rev, B) == std::vector<int32_t>{ 1, 2 }));
    for (const gl::NormKey& key : probeUniverse())
        ASSERT_TRUE(reverseOwners(rev, key) == bruteForceOwners(fwd, key));

    // A wholesale rebuild from the mutated forward map agrees.
    gl::ReverseArgsIndex fresh(&lb);
    fresh.rebuildReverseIndex(fwd, lb);
    for (const gl::NormKey& key : probeUniverse())
        ASSERT_TRUE(reverseOwners(fresh, key) == reverseOwners(rev, key));
}
