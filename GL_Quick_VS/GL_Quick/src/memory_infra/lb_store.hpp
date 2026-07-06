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

#pragma once

#include "global_memory_manager.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <new>
#include <utility>
#include <vector>

namespace gl {

    /// @brief A non-relocating slab allocator for LB node objects (`Memory`),
    ///        carving fixed-size slots out of a never-deloaded pool's blocks
    ///        instead of the malloc heap.
    ///
    /// @details
    /// The statification campaign moves every per-LB allocation off the malloc
    /// heap onto the static memory hierarchy. The last large heap user is the
    /// `Memory` node object itself — historically a `new Memory()` per LB. The
    /// `LbStore` replaces that `new` / `delete` pair: it draws `blockBytes`
    /// blocks from a dedicated `GlobalMemoryManager` (the LB-body pool,
    /// `PoolKind::Lb`) and carves each block into equal, aligned, fixed-size
    /// SLOTS that hold one `Memory` each.
    ///
    /// The store is `Memory`-agnostic: it manages raw slots of a size and
    /// alignment fixed at construction (`sizeof(Memory)` / `alignof(Memory)`,
    /// supplied where the type is complete). `create<T>` placement-news a `T`
    /// into a slot and `destroy<T>` runs the destructor and returns the slot.
    ///
    /// **Non-relocating — the load-bearing property.** A `Memory` is non-movable
    /// and non-copyable (reference members, two `std::atomic` claim words, and
    /// six embedded non-relocatable `LbArena`s), and the rest of the prover
    /// relies on every `Memory*` (`parentMemory`, the `SimpleMapStore` edge
    /// children, the `mailLog` pointer keys) staying valid for
    /// the object's whole life. A slot is therefore allocated once and NEVER
    /// moved; a returned slot is recycled in place. This is why the backing
    /// container is a slab — not a `std::vector<Memory>` (which reallocates) nor
    /// a `PagedVector<Memory>` (which `static_assert`s trivially-copyable `T`).
    ///
    /// **Never deloaded.** The pool is stand-alone: a freed slot is parked on an
    /// intrusive free-list and reused IN PLACE (across batches too) — the store
    /// never returns a block to the pool MID-run, so within a store's life the
    /// pool grows only to the peak simultaneously-live LB count. At teardown
    /// `releaseAll` (in `~LbStore`) returns every block to the pool, mirroring
    /// `LbArena::~LbArena`, so the several `ExpressionAnalyzer`s a process builds
    /// reuse the pool rather than summing their peaks. Nothing reads its grant
    /// ledger, so it is invisible to every deload / throttle / steward decision
    /// (the mail-pool precedent).
    ///
    /// **Off-heap payloads.** The free-list is intrusive — a freed slot stores
    /// the next-free pointer in its own (now raw) bytes — so the slots and the
    /// recycle list cost no heap. The only heap is the block list `blocks_` (one
    /// `char*` per drawn block, needed for `releaseAll`) — the acknowledged
    /// tier-1 pool bookkeeping, exactly like `LbArena::blocks_`, O(blocks) not
    /// O(slots), which a later statification tier moves into the blocks. The
    /// invariant binds element payloads only.
    ///
    /// **Thread-safety.** The main grid is built single-threaded (no LB is born
    /// during the parallel proving phase), but the CE-filter clone path
    /// (`Memory::cloneFactsTemplate`) allocates one LB per conjecture on worker
    /// threads. So `allocateSlot` / `freeSlot` are guarded by an internal mutex;
    /// the underlying `acquireBlock` is itself mutex-guarded by the manager.
    ///
    /// **Failure is an assert (Rule 19).** Pool exhaustion asserts inside the
    /// manager naming `static_lb_pool_bytes`; a slot larger than a block, a
    /// double-free underflow, and a null `destroy` argument each assert here —
    /// never a silent fallback.
    ///
    /// @invariant A slot, once handed out, is never relocated while live; its
    ///            address is stable until `destroy` / `freeSlot`.
    /// @invariant `slotsInUse() == creates - destroys` at every quiescent point.
    /// @see `lbMemory()` (the backing pool), `GlobalMemoryManager::acquireBlock`.
    class LbStore {
    public:
        /// @brief Bind the store to a pool and fix its slot geometry.
        ///
        /// @details
        /// Stores the manager pointer and the slot size / alignment; computes no
        /// geometry and draws no block (lazy — a store that never allocates
        /// consumes zero pool blocks, mirroring `LbArena`). The block size is
        /// read, and the per-block slot count computed, at the first
        /// `allocateSlot` (the manager is initialized by then). `slotBytes` is
        /// rounded up to `slotAlign` for the slot stride.
        ///
        /// @param manager   The dedicated LB-body pool manager (`&lbMemory()` in
        ///                   production; a private instance in tests).
        /// @param slotBytes  Bytes per slot — `sizeof(Memory)` at the production
        ///                   call site (where `Memory` is complete). Must be
        ///                   positive and at least `sizeof(void*)` (the intrusive
        ///                   free-list link).
        /// @param slotAlign  Slot alignment — `alignof(Memory)`. Must be a power
        ///                   of two and at least `alignof(void*)`.
        LbStore(GlobalMemoryManager* manager,
                std::size_t slotBytes,
                std::size_t slotAlign)
            : manager_(manager), slotBytes_(slotBytes), slotAlign_(slotAlign) {
            assert(manager_ != nullptr);
            assert(slotBytes_ >= sizeof(void*));
            assert(slotAlign_ >= alignof(void*));
            assert((slotAlign_ & (slotAlign_ - 1)) == 0
                && "LbStore slot alignment must be a power of two");
        }

        LbStore(const LbStore&) = delete;
        LbStore& operator=(const LbStore&) = delete;

        /// @brief Return every drawn block to the pool (mirrors `~LbArena`).
        ///
        /// @details
        /// Calls `releaseAll`. After `destroyGrid` has destroyed every child LB
        /// the store's blocks hold no live shell, so the return is clean; a
        /// teardown path that skips `destroyGrid` still returns the blocks (the
        /// abandoned shells' own content arenas leak exactly as an un-`delete`d
        /// `new Memory()` would have — no crash, no double-free). The pool
        /// singleton outlives every `LbStore`, so this is safe at process exit.
        ~LbStore();

        /// @brief Hand out a raw, aligned, slot-sized chunk of pool memory.
        ///
        /// @details
        /// Mutex-guarded. Prefers a recycled slot (popped off the intrusive
        /// free-list); else bump-carves the current block; else draws a fresh
        /// block from the pool (the only block traffic). Lazily computes the slot
        /// geometry on first use. Asserts the returned address satisfies the
        /// requested alignment. Pool exhaustion asserts inside `acquireBlock`.
        ///
        /// @return A pointer to `slotBytes` bytes, aligned to `slotAlign`,
        ///         lifetime-stable until the matching `freeSlot`.
        void* allocateSlot();

        /// @brief Return a slot to the store for reuse.
        ///
        /// @details
        /// Mutex-guarded. The slot's raw bytes become the intrusive free-list
        /// node (its first `sizeof(void*)` bytes hold the next-free link). The
        /// block is NOT returned to the pool (never-deloaded; the slot is reused
        /// in place). Asserts a non-null argument and a positive in-use count
        /// (double-free / underflow tripwire).
        ///
        /// @param slot A pointer previously returned by `allocateSlot`.
        void freeSlot(void* slot);

        /// @brief Construct a `T` in a fresh slot and return it (the `new`
        ///        replacement).
        ///
        /// @details
        /// `allocateSlot` + placement-new. Asserts the slot fits `T` (the
        /// geometry was sized for `Memory`; a larger `T` is a programming error,
        /// not a fallback). The returned pointer is address-stable for `T`'s
        /// whole life. The template is instantiated only where `T` is complete
        /// (the call sites), so this header need not include `memory.hpp`.
        ///
        /// @tparam T     The object type (`Memory` in production).
        /// @tparam Args  Constructor argument types.
        /// @param  args  Forwarded to `T`'s constructor.
        /// @return A pointer to the constructed, non-relocatable `T`.
        template <class T, class... Args>
        T* create(Args&&... args) {
            assert(sizeof(T) <= slotBytes_
                && "LbStore slot too small for T — raise the slot size");
            assert(alignof(T) <= slotAlign_
                && "LbStore slot under-aligned for T");
            void* mem = allocateSlot();
            return ::new (mem) T(std::forward<Args>(args)...);
        }

        /// @brief Destroy a `T` and return its slot (the `delete` replacement).
        ///
        /// @details
        /// Runs `T`'s destructor, then `freeSlot`. Asserts a non-null argument
        /// (Rule 19 — the `delete` sites this replaces never delete null; a null
        /// here is a bug to surface, not to swallow).
        ///
        /// @tparam T   The object type (`Memory` in production).
        /// @param  obj A pointer previously returned by `create<T>`.
        template <class T>
        void destroy(T* obj) {
            assert(obj != nullptr && "LbStore::destroy on a null pointer");
            obj->~T();
            freeSlot(obj);
        }

        /// @brief Live slot count (creates minus destroys). Telemetry only.
        /// @return The number of currently-outstanding slots.
        std::size_t slotsInUse() const;

        /// @brief High-water mark of simultaneously-live slots. Telemetry only.
        /// @return The peak `slotsInUse()` observed since construction.
        std::size_t peakSlotsInUse() const;

        /// @brief Number of pool blocks the store currently holds. Telemetry only.
        /// @return The count of blocks drawn from the pool and not yet returned
        ///         (blocks return only at `releaseAll` / teardown).
        std::size_t blocksHeld() const;

        /// @brief Return every drawn block to the pool and reset to empty.
        ///
        /// @details
        /// Mirrors `LbArena::releaseAll`: hands every block in `blocks_` back via
        /// `GlobalMemoryManager::releaseBlock`, clears the block list and the
        /// intrusive free-list, and zeroes the cursors and live count. Legal only
        /// at teardown, when no live shell is read again. Mutex-guarded.
        void releaseAll();

    private:
        /// @brief Intrusive free-list node — overlays a free slot's raw bytes.
        struct FreeSlot {
            FreeSlot* next;
        };

        /// @brief Compute the slot stride and per-block slot count on first use.
        ///
        /// @details
        /// Deferred from the constructor so the store binds before the pool is
        /// initialized (the `ExpressionAnalyzer` member-init / ctor-body order).
        /// Idempotent — recomputes nothing once `slotsPerBlock_` is set. Asserts
        /// at least one slot fits a block (else `sizeof(Memory)` exceeds the
        /// block size — raise `static_lb_block_bytes`). Caller holds `mutex_`.
        void ensureGeometry();

        GlobalMemoryManager* manager_;
        std::size_t slotBytes_;
        std::size_t slotAlign_;
        std::size_t slotStride_ = 0;     ///< slotBytes_ rounded up to slotAlign_
        std::size_t slotsPerBlock_ = 0;  ///< 0 until ensureGeometry runs
        char* currentBlock_ = nullptr;   ///< block being bump-carved
        std::size_t currentCursor_ = 0;  ///< slots carved from currentBlock_
        FreeSlot* freeHead_ = nullptr;   ///< intrusive recycle list
        std::size_t slotsInUse_ = 0;
        std::size_t peakSlotsInUse_ = 0;
        std::vector<char*> blocks_;      ///< every drawn block, for releaseAll
        mutable std::mutex mutex_;
    };

}
