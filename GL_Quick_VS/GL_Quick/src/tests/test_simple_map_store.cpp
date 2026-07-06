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

#include "test_harness.hpp"
#include "../simple_map_store.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace {

    // SimpleMapStore uses `const Memory*` parents purely as map-key identity and
    // never dereferences a parent or a child (findChild returns the stored child
    // pointer; the test compares addresses), so fabricated addresses stand in for
    // LBs — keeping the test free of the arena-backed `Memory` constructor.
    gl::Memory* fakeLb(std::uintptr_t id) {
        return reinterpret_cast<gl::Memory*>(id);
    }

    // A self-contained store arena on a PRIVATE pool for the edges + heads. The
    // routing-key strings intern into the process-wide skeletonInterner() the test
    // harness sets up at startup (on lbMemory()), so no local interner setup is
    // needed. Default config: 1 MiB pool / 256 KiB block (default page).
    struct StoreFixture {
        gl::GlobalMemoryManager mem;
        gl::LbArena arena;
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::SimpleMapStore store;
        explicit StoreFixture(
            gl::StaticMemoryConfig cfg = gl::StaticMemoryConfig{ 1 << 20, 1 << 18 })
            : mem(), arena(&mem), dirty(gl::DirtyState::Clean), store(&arena, &dirty) {
            mem.init(cfg);
        }
    };

}  // namespace

// lbKey casts the Memory* bit pattern faithfully (pure identity, never deref'd).
TEST(simple_map_store, lbkey_is_pointer_identity) {
    ASSERT_EQ(gl::SimpleMapStore::lbKey(fakeLb(0x1234)),
              static_cast<std::int64_t>(0x1234));
    ASSERT_EQ(gl::SimpleMapStore::lbKey(fakeLb(0xABCD)),
              static_cast<std::int64_t>(0xABCD));
}

// A linked child is found under its routing key; distinct keys resolve distinctly.
TEST(simple_map_store, multiple_children_under_one_parent) {
    StoreFixture f;
    gl::SimpleMapStore& s = f.store;
    gl::Memory* p = fakeLb(0x100);
    gl::Memory* c1 = fakeLb(0xA);
    gl::Memory* c2 = fakeLb(0xB);
    gl::Memory* c3 = fakeLb(0xC);
    s.linkChild(p, "(k1)", c1);
    s.linkChild(p, "(k2)", c2);
    s.linkChild(p, "(k3)", c3);
    ASSERT_EQ(s.findChild(p, "(k1)"), c1);
    ASSERT_EQ(s.findChild(p, "(k2)"), c2);
    ASSERT_EQ(s.findChild(p, "(k3)"), c3);
}

// An absent edge is nullptr — a defined query result, the find==end() twin.
TEST(simple_map_store, find_absent_is_null) {
    StoreFixture f;
    gl::SimpleMapStore& s = f.store;
    gl::Memory* p = fakeLb(0x100);
    ASSERT_TRUE(s.findChild(p, "(never-linked)") == nullptr);   // parent has no edges
    s.linkChild(p, "(k1)", fakeLb(0xA));
    ASSERT_TRUE(s.findChild(p, "(k2)") == nullptr);             // present parent, other key
    ASSERT_TRUE(s.findChild(fakeLb(0x999), "(k1)") == nullptr); // different parent
}

// The same routing key under two parents yields two independent edges — the chain
// is keyed by (parent, key), not by the shared interner id alone.
TEST(simple_map_store, same_key_distinct_parents) {
    StoreFixture f;
    gl::SimpleMapStore& s = f.store;
    gl::Memory* pA = fakeLb(0x100);
    gl::Memory* pB = fakeLb(0x200);
    gl::Memory* childA = fakeLb(0xA);
    gl::Memory* childB = fakeLb(0xB);
    s.linkChild(pA, "(in2[rec,1,s])", childA);
    s.linkChild(pB, "(in2[rec,1,s])", childB);   // same key, different parent
    ASSERT_EQ(s.findChild(pA, "(in2[rec,1,s])"), childA);
    ASSERT_EQ(s.findChild(pB, "(in2[rec,1,s])"), childB);
}

// Inserts interleaved across parents keep each parent's chain independent (the
// per-parent back-link, the property a flat append-only CSR could not give).
TEST(simple_map_store, interleaved_inserts_keep_chains_independent) {
    StoreFixture f;
    gl::SimpleMapStore& s = f.store;
    gl::Memory* pA = fakeLb(0x100);
    gl::Memory* pB = fakeLb(0x200);
    gl::Memory* a1 = fakeLb(0xA1);
    gl::Memory* a2 = fakeLb(0xA2);
    gl::Memory* b1 = fakeLb(0xB1);
    s.linkChild(pA, "(k1)", a1);
    s.linkChild(pB, "(k1)", b1);   // B's insert lands between A's two
    s.linkChild(pA, "(k2)", a2);
    ASSERT_EQ(s.findChild(pA, "(k1)"), a1);
    ASSERT_EQ(s.findChild(pA, "(k2)"), a2);
    ASSERT_EQ(s.findChild(pB, "(k1)"), b1);
    int countA = 0;
    s.forEachChild(pA, [&](const gl::StrSpan&, gl::Memory*) { ++countA; });
    int countB = 0;
    s.forEachChild(pB, [&](const gl::StrSpan&, gl::Memory*) { ++countB; });
    ASSERT_EQ(countA, 2);
    ASSERT_EQ(countB, 1);
}

// forEachChild yields children sorted by DECODED routing-key string — byte-
// identical to the old std::map<std::string, Memory*> iteration order, regardless
// of insertion order. This is the determinism pin of the migration.
TEST(simple_map_store, foreach_yields_decoded_key_order) {
    StoreFixture f;
    gl::SimpleMapStore& s = f.store;
    gl::Memory* parent = fakeLb(0x100);
    gl::Memory* ca = fakeLb(0xA);
    gl::Memory* cb = fakeLb(0xB);
    gl::Memory* cc = fakeLb(0xC);
    s.linkChild(parent, "(gamma)", cc);   // inserted out of sorted order
    s.linkChild(parent, "(alpha)", ca);
    s.linkChild(parent, "(beta)", cb);
    std::vector<std::string> keys;
    std::vector<gl::Memory*> kids;
    s.forEachChild(parent, [&](const gl::StrSpan& k, gl::Memory* c) {
        keys.push_back(std::string(k.ptr, static_cast<std::size_t>(k.len)));
        kids.push_back(c);
    });
    ASSERT_EQ(keys.size(), static_cast<std::size_t>(3));
    ASSERT_TRUE(keys[0] == "(alpha)");   // std::map order: (alpha) < (beta) < (gamma)
    ASSERT_TRUE(keys[1] == "(beta)");
    ASSERT_TRUE(keys[2] == "(gamma)");
    ASSERT_EQ(kids[0], ca);
    ASSERT_EQ(kids[1], cb);
    ASSERT_EQ(kids[2], cc);
}

// forEachChild on a parent with no edges is a defined no-op.
TEST(simple_map_store, foreach_empty_parent_is_noop) {
    StoreFixture f;
    gl::SimpleMapStore& s = f.store;
    int n = 0;
    s.forEachChild(fakeLb(0x100), [&](const gl::StrSpan&, gl::Memory*) { ++n; });
    ASSERT_EQ(n, 0);
}

// clear() drops every edge and head; the store reads empty and finds nothing.
TEST(simple_map_store, clear_resets) {
    StoreFixture f;
    gl::SimpleMapStore& s = f.store;
    gl::Memory* p = fakeLb(0x100);
    s.linkChild(p, "(k)", fakeLb(0xA));
    ASSERT_FALSE(s.empty());
    ASSERT_TRUE(s.findChild(p, "(k)") != nullptr);
    s.clear();
    ASSERT_TRUE(s.empty());
    ASSERT_TRUE(s.findChild(p, "(k)") == nullptr);
}

// Regression guard mirroring the mail-log spill test: many distinct parent keys
// push the heads index into the two-level paged-hash directory (the path that
// silently corrupted neighbouring pages before the fix). At the default 8 KiB page
// (dirCap = 2048) 2500 parents spill, and every findChild must still resolve.
TEST(simple_map_store, forced_two_level_spill_no_corruption) {
    StoreFixture f(gl::StaticMemoryConfig{ 1 << 24, 1 << 18 });   // 16 MiB / 256 KiB
    gl::SimpleMapStore& s = f.store;
    const int N = 2500;   // > dirCap (8 KiB page / 4 = 2048)
    auto parent = [](int i) {
        return fakeLb(0x100000 + static_cast<std::uintptr_t>(i) * 64);
    };
    auto child = [](int i) {
        return fakeLb(0x1 + static_cast<std::uintptr_t>(i));
    };
    for (int i = 0; i < N; ++i) s.linkChild(parent(i), "(child)", child(i));
    for (int i = 0; i < N; ++i) {
        ASSERT_EQ(s.findChild(parent(i), "(child)"), child(i));
        ASSERT_TRUE(s.findChild(parent(i), "(absent)") == nullptr);
    }
}
