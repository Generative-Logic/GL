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
/// @brief Unit tests for `ColdMail` — the per-LB internal-mail mailbox on the
///        deloadable arena (id-form columns).
///
/// @details
/// Exercises the three id-form columns in isolation on a private
/// `GlobalMemoryManager` + cold `LbArena`, with RAW interner ids (this low-level
/// test has no `NameMap` / `originInterner`): statement multiplicity (the whole
/// key incl. levels), an origins blob round trip, the `disintegrationSignals`
/// pack/unpack + read door, `clear`, and the scope-targeted `filterStatements`
/// sweep. The string-boundary wrappers (`insertInternalStatement` /
/// `addInternalMailOrigin` / `setInternalDisintegrationSignal` / `makeHeapMail`)
/// — which intern through a real LB's `NameMap` / `originInterner` — are tested in
/// `test_memory.cpp`; the deload round trip is in `test_lb_deload.cpp`.

#include "test_harness.hpp"

#include "../memory_infra/cold_mail.hpp"

#include <cstdint>
#include <set>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block — the bump-arena substrate this branch uses.
    const gl::StaticMemoryConfig kColdMailCfg{ 1 << 20, 1 << 18 };

    // Pack an (originalId, validityId) pair the way packOriginKey does — the
    // origins / disintegrationSignals key form (memory.hpp is not included here,
    // so the helper is reproduced locally).
    int64_t pk(int32_t o, int32_t v) {
        return static_cast<int64_t>(
            (static_cast<uint64_t>(static_cast<uint32_t>(o)) << 32)
            | static_cast<uint64_t>(static_cast<uint32_t>(v)));
    }
}

// Whole key incl. levels: same (orig,valid) with different level sets stays
// distinct; the exact same triple collapses.
TEST(cold_mail, statement_multiplicity) {
    gl::GlobalMemoryManager g;
    g.init(kColdMailCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdMail cm(&lb, &d);

    cm.insertStatement(1, 16, std::set<int>{ 1, 2 });
    cm.insertStatement(1, 16, std::set<int>{ 3 });
    cm.insertStatement(1, 16, std::set<int>{ 1, 2 });   // exact dup
    cm.insertStatement(2, 20, std::set<int>{ 1 });
    ASSERT_EQ(cm.statements_.count(), 3);               // dup collapsed

    // Collect the decoded id-form keys and check membership (id-insertion order,
    // not string order — the absorb sorts after decoding to strings).
    bool sawA12 = false, sawA3 = false, sawB1 = false;
    for (int32_t id = 1; id <= cm.statements_.count(); ++id) {
        const gl::IntMailStatementKey k = cm.statements_.decodeKey(id);
        if (k.originalId == 1 && k.validityId == 16
            && k.levels == std::vector<int32_t>{ 1, 2 }) sawA12 = true;
        if (k.originalId == 1 && k.validityId == 16
            && k.levels == std::vector<int32_t>{ 3 }) sawA3 = true;
        if (k.originalId == 2 && k.validityId == 20
            && k.levels == std::vector<int32_t>{ 1 }) sawB1 = true;
    }
    ASSERT_TRUE(sawA12 && sawA3 && sawB1);
}

// origins blob round trip: assignRun stores an IntMailOrigin run; recordsAt
// decodes it byte-identically (tag byte + packed dep keys).
TEST(cold_mail, origins_blob_round_trip) {
    gl::GlobalMemoryManager g;
    g.init(kColdMailCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdMail cm(&lb, &d);

    const int64_t key = pk(5, 16);
    const gl::IntMailOrigin r0{ 6 /*tag*/, std::vector<int64_t>{ pk(7, 16) } };
    const gl::IntMailOrigin r1{ 9 /*tag*/,
        std::vector<int64_t>{ pk(8, 16), pk(9, 20) } };
    cm.origins_.assignRun(key, std::vector<gl::IntMailOrigin>{ r0, r1 });

    const int32_t id = cm.origins_.lookup(key);
    ASSERT_TRUE(id != 0);
    const std::vector<gl::IntMailOrigin> got = cm.origins_.recordsAt(id);
    ASSERT_EQ(static_cast<int>(got.size()), 2);
    ASSERT_TRUE(got[0] == r0);
    ASSERT_TRUE(got[1] == r1);
}

// disintegrationSignals: the two bools pack into one byte; upsert overwrites in
// place; the read door unpacks per key, a miss returns {false, false}.
TEST(cold_mail, disintegration_signals_pack_unpack_overwrite_read) {
    gl::GlobalMemoryManager g;
    g.init(kColdMailCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdMail cm(&lb, &d);

    cm.setDisintegrationSignal(pk(26, 16), /*dnd=*/false, /*aod=*/true);
    cm.setDisintegrationSignal(pk(1, 16), /*dnd=*/true, /*aod=*/false);
    cm.setDisintegrationSignal(pk(1, 16), /*dnd=*/true, /*aod=*/true);  // overwrite
    ASSERT_EQ(cm.disintegrationSignals_.count(), 2);                    // upsert

    const std::pair<bool, bool> sa = cm.getDisintegrationSignal(pk(1, 16));
    ASSERT_TRUE(sa.first == true && sa.second == true);                 // overwritten
    const std::pair<bool, bool> sz = cm.getDisintegrationSignal(pk(26, 16));
    ASSERT_TRUE(sz.first == false && sz.second == true);
    // Miss: an unset key returns {false, false}.
    const std::pair<bool, bool> miss = cm.getDisintegrationSignal(pk(99, 16));
    ASSERT_TRUE(miss.first == false && miss.second == false);
}

// The out-param overload of getDisintegrationSignal is bit-identical to the
// pair-returning oracle for both a present key (each of the four bit patterns)
// and an absent key. The pair form is the oracle; the out-param door is the
// heap-free form the standardProcessing drain calls.
TEST(cold_mail, disintegration_signal_out_param_matches_pair) {
    gl::GlobalMemoryManager g;
    g.init(kColdMailCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdMail cm(&lb, &d);

    // Four keys, one per (dnd, aod) bit pattern, plus one deliberately-unset key.
    cm.setDisintegrationSignal(pk(1, 16), /*dnd=*/false, /*aod=*/false);
    cm.setDisintegrationSignal(pk(2, 16), /*dnd=*/false, /*aod=*/true);
    cm.setDisintegrationSignal(pk(3, 16), /*dnd=*/true, /*aod=*/false);
    cm.setDisintegrationSignal(pk(4, 16), /*dnd=*/true, /*aod=*/true);

    const int64_t keys[] = { pk(1, 16), pk(2, 16), pk(3, 16), pk(4, 16),
                             pk(99, 16) /*absent*/ };
    for (const int64_t key : keys) {
        const std::pair<bool, bool> oracle = cm.getDisintegrationSignal(key);
        bool dnd = true, aod = true;  // seed opposite of the absent default
        cm.getDisintegrationSignal(key, dnd, aod);
        ASSERT_TRUE(dnd == oracle.first);
        ASSERT_TRUE(aod == oracle.second);
    }
}

// clear() empties all three columns; empty() reflects it.
TEST(cold_mail, clear_empties_all_columns) {
    gl::GlobalMemoryManager g;
    g.init(kColdMailCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdMail cm(&lb, &d);

    cm.insertStatement(1, 16, std::set<int>{ 1 });
    cm.origins_.assignRun(pk(1, 16), std::vector<gl::IntMailOrigin>{
        gl::IntMailOrigin{ 6, std::vector<int64_t>{ pk(2, 16) } } });
    cm.setDisintegrationSignal(pk(1, 16), true, true);
    ASSERT_TRUE(!cm.empty());

    cm.clear();
    ASSERT_TRUE(cm.empty());
    ASSERT_EQ(cm.statements_.count(), 0);
    ASSERT_EQ(cm.origins_.count(), 0);
    ASSERT_EQ(cm.disintegrationSignals_.count(), 0);
}

// filterStatements sweeps only the statements column, by predicate over the
// id-form key (the wipeSubtree scope sweep); origins untouched.
TEST(cold_mail, filter_statements_scope_sweep) {
    gl::GlobalMemoryManager g;
    g.init(kColdMailCfg);
    gl::LbArena lb(&g);
    gl::DirtyState d = gl::DirtyState::Clean;
    gl::ColdMail cm(&lb, &d);

    cm.insertStatement(1, 16, std::set<int>{ 1 });   // validity 16 (e.g. main)
    cm.insertStatement(2, 20, std::set<int>{ 1 });   // validity 20 (e.g. branchA)
    cm.insertStatement(3, 20, std::set<int>{ 2 });   // validity 20
    cm.origins_.assignRun(pk(2, 20), std::vector<gl::IntMailOrigin>{
        gl::IntMailOrigin{ 6, std::vector<int64_t>{ pk(4, 20) } } });

    const int removed = cm.filterStatements(
        [](const gl::IntMailStatementKey& k) { return k.validityId == 20; });
    ASSERT_EQ(removed, 2);
    ASSERT_EQ(cm.statements_.count(), 1);
    const gl::IntMailStatementKey kept = cm.statements_.decodeKey(1);
    ASSERT_TRUE(kept.originalId == 1 && kept.validityId == 16);
    ASSERT_EQ(cm.origins_.count(), 1);               // origins untouched
}
