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
/// @brief Unit tests for the LB object store (`LbStore`, the non-relocating
///        slab allocator that holds `Memory` node shells off the malloc heap).
///
/// @details
/// Covers create/destroy running the constructor and destructor and tracking
/// the live count, intrusive-free-list LIFO slot reuse, slot alignment, growth
/// across multiple pool blocks with peak/blocks-held telemetry, and address
/// stability of live objects across further allocations.

#include "test_harness.hpp"

#include "../memory_infra/global_memory_manager.hpp"
#include "../memory_infra/lb_store.hpp"

#include <cstddef>
#include <cstdint>
#include <set>
#include <vector>

namespace {
    // 32 KiB pool / 8 KiB block — 4 blocks. Page defaults to 8 KiB (== block),
    // satisfying the carving contract. PoolKind::Lb for fidelity (kind is
    // irrelevant to the store, which only calls acquireBlock).
    const gl::StaticMemoryConfig kLbStoreCfg{ 32768, 8192, 8192,
                                              gl::PoolKind::Lb };

    // Memory-stand-in: a constructed/destructed object whose live count the
    // store must keep correct. Smaller than the slot (the store is sized for
    // sizeof(Memory)); the test slots are 64 B / 16 B-aligned to mimic that.
    struct Widget {
        int id;
        static int liveCount;
        explicit Widget(int i) : id(i) { ++liveCount; }
        ~Widget() { --liveCount; }
    };
    int Widget::liveCount = 0;

    constexpr std::size_t kSlotBytes = 64;
    constexpr std::size_t kSlotAlign = 16;
}

TEST(lb_store, create_destroy_tracks_live_count) {
    Widget::liveCount = 0;
    gl::GlobalMemoryManager g;
    g.init(kLbStoreCfg);
    gl::LbStore store(&g, kSlotBytes, kSlotAlign);

    std::vector<Widget*> live;
    for (int i = 0; i < 10; ++i) {
        live.push_back(store.create<Widget>(i));
    }
    ASSERT_EQ(Widget::liveCount, 10);
    ASSERT_EQ(store.slotsInUse(), static_cast<std::size_t>(10));
    for (int i = 0; i < 10; ++i) {
        ASSERT_EQ(live[static_cast<std::size_t>(i)]->id, i);
    }

    for (Widget* w : live) {
        store.destroy(w);
    }
    ASSERT_EQ(Widget::liveCount, 0);
    ASSERT_EQ(store.slotsInUse(), static_cast<std::size_t>(0));
}

TEST(lb_store, free_list_reuses_slots_lifo) {
    gl::GlobalMemoryManager g;
    g.init(kLbStoreCfg);
    gl::LbStore store(&g, kSlotBytes, kSlotAlign);

    void* a = store.allocateSlot();
    void* b = store.allocateSlot();
    ASSERT_TRUE(a != b);
    // Free a then b; the intrusive free-list is LIFO, so the next two allocs
    // return b then a.
    store.freeSlot(a);
    store.freeSlot(b);
    void* first = store.allocateSlot();
    void* second = store.allocateSlot();
    ASSERT_TRUE(first == b);
    ASSERT_TRUE(second == a);
    ASSERT_EQ(store.slotsInUse(), static_cast<std::size_t>(2));
}

TEST(lb_store, slots_are_aligned) {
    gl::GlobalMemoryManager g;
    g.init(kLbStoreCfg);
    gl::LbStore store(&g, kSlotBytes, kSlotAlign);

    for (int i = 0; i < 200; ++i) {
        void* p = store.allocateSlot();
        ASSERT_EQ(reinterpret_cast<std::uintptr_t>(p) % kSlotAlign,
                  static_cast<std::uintptr_t>(0));
    }
}

TEST(lb_store, grows_across_blocks_and_tracks_peak) {
    gl::GlobalMemoryManager g;
    g.init(kLbStoreCfg);
    gl::LbStore store(&g, kSlotBytes, kSlotAlign);

    // 8 KiB block / 64 B stride = 128 slots per block. 129 allocations span a
    // second block.
    std::vector<void*> slots;
    for (int i = 0; i < 129; ++i) {
        slots.push_back(store.allocateSlot());
    }
    ASSERT_EQ(store.blocksHeld(), static_cast<std::size_t>(2));
    ASSERT_EQ(store.peakSlotsInUse(), static_cast<std::size_t>(129));

    // All addresses are distinct (no slot overlap).
    std::set<void*> distinct(slots.begin(), slots.end());
    ASSERT_EQ(distinct.size(), static_cast<std::size_t>(129));

    // Freeing then re-allocating reuses slots without drawing new blocks; peak
    // stays at the high-water mark.
    for (void* p : slots) {
        store.freeSlot(p);
    }
    ASSERT_EQ(store.slotsInUse(), static_cast<std::size_t>(0));
    for (int i = 0; i < 50; ++i) {
        store.allocateSlot();
    }
    ASSERT_EQ(store.blocksHeld(), static_cast<std::size_t>(2));
    ASSERT_EQ(store.peakSlotsInUse(), static_cast<std::size_t>(129));
}

TEST(lb_store, live_objects_are_address_stable) {
    Widget::liveCount = 0;
    gl::GlobalMemoryManager g;
    g.init(kLbStoreCfg);
    gl::LbStore store(&g, kSlotBytes, kSlotAlign);

    Widget* w0 = store.create<Widget>(100);
    Widget* w1 = store.create<Widget>(101);
    const Widget* a0 = w0;
    const Widget* a1 = w1;

    // Allocating more must not relocate the earlier objects.
    for (int i = 0; i < 300; ++i) {
        store.create<Widget>(i);
    }
    ASSERT_TRUE(w0 == a0);
    ASSERT_TRUE(w1 == a1);
    ASSERT_EQ(w0->id, 100);
    ASSERT_EQ(w1->id, 101);
}

TEST(lb_store, release_all_returns_blocks_to_pool) {
    gl::GlobalMemoryManager g;
    g.init(kLbStoreCfg);
    gl::LbStore store(&g, kSlotBytes, kSlotAlign);

    // Draw enough slots to span two blocks (128 slots / block).
    for (int i = 0; i < 200; ++i) {
        store.allocateSlot();
    }
    ASSERT_EQ(store.blocksHeld(), static_cast<std::size_t>(2));
    ASSERT_EQ(g.blocksInUse(), static_cast<std::int64_t>(2));

    store.releaseAll();
    ASSERT_EQ(store.blocksHeld(), static_cast<std::size_t>(0));
    ASSERT_EQ(store.slotsInUse(), static_cast<std::size_t>(0));
    ASSERT_EQ(g.blocksInUse(), static_cast<std::int64_t>(0));

    // Reusable after a release (cached geometry).
    void* p = store.allocateSlot();
    ASSERT_TRUE(p != nullptr);
    ASSERT_EQ(store.blocksHeld(), static_cast<std::size_t>(1));
}

TEST(lb_store, destructor_returns_blocks_to_pool) {
    gl::GlobalMemoryManager g;
    g.init(kLbStoreCfg);
    {
        gl::LbStore store(&g, kSlotBytes, kSlotAlign);
        for (int i = 0; i < 200; ++i) {
            store.allocateSlot();
        }
        ASSERT_EQ(g.blocksInUse(), static_cast<std::int64_t>(2));
    }
    // ~LbStore returned every block to the pool (mirrors ~LbArena).
    ASSERT_EQ(g.blocksInUse(), static_cast<std::int64_t>(0));
}
