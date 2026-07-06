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

#include "lb_arena.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace gl {

    /// @brief Fixed-capacity paged `int32` slot array on the LB arena — the
    ///        THROW-AWAY, MUTABLE-slot twin of `PagedVector`, the home of a hot
    ///        open-addressing hash index.
    ///
    /// @details
    /// `PagedVector` is const-access by design: it has no mutable `operator[]`,
    /// so the deload skip-unchanged contract (every mutation routes through a
    /// dirty-marking append) cannot be bypassed. An open-addressing hash index
    /// needs the opposite — scattered IN-PLACE slot writes — and it is DERIVED
    /// (rebuilt from the keys, never serialized), so it does not belong in the
    /// deload image at all. `PagedHashIndex` is the separate container that
    /// serves exactly that: a flat array of `int32` slots packed on arena pages,
    /// with `at` / `set` for in-place reads and writes, allocated wholesale by
    /// `reset` and freed by `clear`.
    ///
    /// **Throw-away static.** Its pages come from the LB arena's page tier
    /// (`allocPage` / `pageAt` / `freePage`) — pool-backed, NEVER the heap — but
    /// it is transient: it is not in `LbMemory::visitContainers`, is never
    /// deloaded, carries no dirty flag, and is rebuilt on reload. So it has the
    /// no-malloc property of the cold containers without their persistence cost.
    /// Pages are addressed by stable virtual id (vid) and resolved through
    /// `pageAt` per access, so the background compaction may rebind a vid without
    /// this container noticing (I-107).
    ///
    /// Slots pack power-of-two-per-page (one shift + one mask to split a slot
    /// index into page + offset), like `PagedVector`; the page directory lives in
    /// the arena, never on the heap, and is adaptive: an inline vid for one page,
    /// one directory page for `2 .. slotsPerPage` data pages, then a TWO-LEVEL
    /// directory (an L2 root page of level-1 directory-page vids; each L1 page
    /// holds up to `slotsPerPage` data-page vids) for larger arrays — so the
    /// index itself spans several pages. Both the slot and the directory fan-out
    /// are `int32`, so `slotsPerPage == dirCap` and the same `shift_` / `mask_`
    /// split the slot-into-page AND the page-into-(L1, slot). Capacity is fixed
    /// per `reset` — the hash table rebuilds wholesale on growth, so there is no
    /// incremental promotion/demotion ladder; `reset` builds the whole shape.
    /// Overflowing two levels (`slotsPerPage²` pages) HARD-ASSERTS, naming the
    /// three-level directory — replacing the former unchecked write that silently
    /// ran past the single directory page (D-144).
    ///
    /// @invariant Between `reset(cap)` and the next `reset` / `clear`, slot
    ///            indices `[0, cap)` are readable and writable; a freshly
    ///            `reset` array reads all-zero.
    /// @see `PagedVector` (the persistent const-access sibling), `LbArena`,
    ///      `ColdHashSet`, D-166.
    class PagedHashIndex {
    public:
        /// @brief Bind to the owning LB's arena (the page source).
        ///
        /// @details
        /// Lazy like `PagedVector`: construction touches neither the arena nor
        /// `pageBytes()` (the pool is not initialised at member-construction
        /// time), so an unused container consumes no pages. No dirty flag — the
        /// container is throw-away and never deloaded.
        ///
        /// @param arena The LB's page-dispensing arena. Must be non-null.
        explicit PagedHashIndex(LbArena* arena) : arena_(arena) {
            assert(arena_ != nullptr);
        }

        PagedHashIndex(const PagedHashIndex&) = delete;
        PagedHashIndex& operator=(const PagedHashIndex&) = delete;
        PagedHashIndex(PagedHashIndex&&) = delete;
        PagedHashIndex& operator=(PagedHashIndex&&) = delete;

        /// @brief Frees every held page back to the arena.
        ~PagedHashIndex() { clear(); }

        /// @brief Slot capacity (0 before the first `reset`).
        ///
        /// @return The number of slots; a power of two when non-zero.
        int32_t capacity() const { return capacity_; }

        /// @brief Whether the index holds no slots.
        ///
        /// @return `true` when `capacity() == 0`.
        bool empty() const { return capacity_ == 0; }

        /// @brief Read slot `i`.
        ///
        /// @param i Slot index in `[0, capacity())`.
        /// @return The stored `int32` (0 = empty for the hash-index use).
        int32_t at(int32_t i) const {
            assert(i >= 0 && i < capacity_ && "PagedHashIndex::at out of range");
            return *reinterpret_cast<const int32_t*>(slotPtr(i));
        }

        /// @brief Write slot `i` IN PLACE — the operation `PagedVector` forbids.
        ///
        /// @details
        /// Safe precisely because this container is throw-away: it is never part
        /// of a deload image, so an in-place write that does not mark any dirty
        /// flag cannot corrupt the skip-unchanged contract.
        ///
        /// @param i Slot index in `[0, capacity())`.
        /// @param v The value to store.
        void set(int32_t i, int32_t v) {
            assert(i >= 0 && i < capacity_ && "PagedHashIndex::set out of range");
            *reinterpret_cast<int32_t*>(slotPtr(i)) = v;
        }

        /// @brief Free any held pages, then allocate and ZERO `cap` slots — the
        ///        throw-away rebuild entry (the hash index sizes here on growth).
        ///
        /// @details
        /// Allocates `ceil(cap / slotsPerPage)` data pages up front and memsets
        /// each to 0 (a reused page may carry stale bytes), so the whole array
        /// reads all-zero. `cap` should be a power of two (the caller sizes it
        /// so); the power-of-two slots-per-page split then indexes by shift +
        /// mask. Builds the whole directory shape in one shot — inline, single
        /// directory page, or (for a large array) the two-level directory whose
        /// L2 root points at level-1 directory pages. Overflowing two levels
        /// (`slotsPerPage²` data pages) HARD-ASSERTS — the bounds check the
        /// former single-directory loop lacked, where `dir[p]` past `slotsPerPage`
        /// silently wrote off the directory page and corrupted the neighbouring
        /// arena page (D-144).
        ///
        /// @param cap Slot capacity to allocate; `>= 0`.
        void reset(int32_t cap) {
            clear();
            assert(cap >= 0);
            if (cap == 0) return;
            ensureGeometry();
            capacity_ = cap;
            numPages_ = (cap + slotsPerPage_ - 1) >> shift_;   // ceil
            const std::size_t pageBytes =
                static_cast<std::size_t>(arena_->pageBytes());
            const int32_t dirCap = slotsPerPage_;   // int32 directory entry == int32 slot
            if (numPages_ == 1) {                   // single data page, no directory
                rootVid_ = arena_->allocPage();
                std::memset(arena_->pageAt(rootVid_), 0, pageBytes);
            } else if (numPages_ <= dirCap) {       // one directory page
                rootVid_ = arena_->allocPage();
                int32_t* dir =
                    reinterpret_cast<int32_t*>(arena_->pageAt(rootVid_));
                for (int32_t p = 0; p < numPages_; ++p) {
                    const int32_t vid = arena_->allocPage();
                    dir[p] = vid;
                    std::memset(arena_->pageAt(vid), 0, pageBytes);
                }
            } else {                                // two-level directory
                assert(static_cast<int64_t>(numPages_)
                           <= static_cast<int64_t>(dirCap) * dirCap
                    && "PagedHashIndex exceeded the two-level directory — add "
                       "the three-level directory for a larger hash index");
                rootVid_ = arena_->allocPage();     // the L2 root page
                int32_t* l2 =
                    reinterpret_cast<int32_t*>(arena_->pageAt(rootVid_));
                const int32_t l1Count = (numPages_ + dirCap - 1) >> shift_;  // ceil
                int32_t made = 0;
                for (int32_t li = 0; li < l1Count; ++li) {
                    const int32_t l1vid = arena_->allocPage();
                    l2[li] = l1vid;
                    int32_t* l1 =
                        reinterpret_cast<int32_t*>(arena_->pageAt(l1vid));
                    const int32_t inThis = std::min(dirCap, numPages_ - made);
                    for (int32_t s = 0; s < inThis; ++s) {
                        const int32_t vid = arena_->allocPage();
                        l1[s] = vid;
                        std::memset(arena_->pageAt(vid), 0, pageBytes);
                        ++made;
                    }
                }
                arena_->recordTwoLevelPromotion(numPages_);   // telemetry only
            }
        }

        /// @brief Free every held page (data, level-1 directory, and L2 root,
        ///        per the directory shape) and reset to empty.
        void clear() {
            if (capacity_ == 0) return;
            const int32_t dirCap = slotsPerPage_;
            if (numPages_ == 1) {
                arena_->freePage(rootVid_);
            } else if (numPages_ <= dirCap) {     // single directory page
                const int32_t* dir =
                    reinterpret_cast<const int32_t*>(arena_->pageAt(rootVid_));
                for (int32_t p = 0; p < numPages_; ++p)
                    arena_->freePage(dir[p]);
                arena_->freePage(rootVid_);   // the directory page itself
            } else {                              // two-level directory
                const int32_t* l2 =
                    reinterpret_cast<const int32_t*>(arena_->pageAt(rootVid_));
                const int32_t l1Count = (numPages_ + dirCap - 1) >> shift_;
                for (int32_t li = 0; li < l1Count; ++li) {
                    const int32_t* l1 = reinterpret_cast<const int32_t*>(
                        arena_->pageAt(l2[li]));
                    const int32_t base = li << shift_;
                    const int32_t inThis = std::min(dirCap, numPages_ - base);
                    for (int32_t s = 0; s < inThis; ++s)
                        arena_->freePage(l1[s]);
                    arena_->freePage(l2[li]);     // the level-1 directory page
                }
                arena_->freePage(rootVid_);       // the L2 root
            }
            rootVid_ = kNoVid;
            capacity_ = 0;
            numPages_ = 0;
        }

        /// @brief Live page footprint in bytes (the steward's pressure input —
        ///        the throw-away pages are real arena pages while resident).
        ///
        /// @return Held-page bytes; 0 when empty.
        int64_t liveBytes() const {
            if (numPages_ == 0) return 0;
            const int32_t dirCap = slotsPerPage_;
            int64_t held = numPages_;                       // data pages
            if (numPages_ > 1 && numPages_ <= dirCap) {
                held += 1;                                  // one directory page
            } else if (numPages_ > dirCap) {
                const int64_t l1Count =
                    (static_cast<int64_t>(numPages_) + dirCap - 1) >> shift_;
                held += l1Count + 1;                        // L1 pages + L2 root
            }
            return held * static_cast<int64_t>(arena_->pageBytes());
        }

    private:
        /// @brief Sentinel "no vid" for the empty / unbound directory root.
        static constexpr int32_t kNoVid = -1;

        /// @brief Byte address of slot `i` (one shift + one mask + one resolve).
        ///
        /// @param i Slot index in `[0, capacity())`.
        /// @return Writable byte pointer to the 4-byte slot.
        char* slotPtr(int32_t i) {
            const int32_t pageIdx = i >> shift_;
            int32_t vid;
            if (numPages_ == 1) {
                vid = rootVid_;
            } else if (numPages_ <= mask_ + 1) {        // single directory page
                vid = reinterpret_cast<const int32_t*>(
                          arena_->pageAt(rootVid_))[pageIdx];
            } else {                                     // two-level directory
                const int32_t l1vid = reinterpret_cast<const int32_t*>(
                    arena_->pageAt(rootVid_))[pageIdx >> shift_];
                vid = reinterpret_cast<const int32_t*>(
                    arena_->pageAt(l1vid))[pageIdx & mask_];
            }
            return arena_->pageAt(vid)
                 + static_cast<std::size_t>(i & mask_) * sizeof(int32_t);
        }

        /// @brief Const overload of `slotPtr`.
        ///
        /// @param i Slot index in `[0, capacity())`.
        /// @return Read-only byte pointer to the 4-byte slot.
        const char* slotPtr(int32_t i) const {
            const int32_t pageIdx = i >> shift_;
            int32_t vid;
            if (numPages_ == 1) {
                vid = rootVid_;
            } else if (numPages_ <= mask_ + 1) {        // single directory page
                vid = reinterpret_cast<const int32_t*>(
                          arena_->pageAt(rootVid_))[pageIdx];
            } else {                                     // two-level directory
                const int32_t l1vid = reinterpret_cast<const int32_t*>(
                    arena_->pageAt(rootVid_))[pageIdx >> shift_];
                vid = reinterpret_cast<const int32_t*>(
                    arena_->pageAt(l1vid))[pageIdx & mask_];
            }
            return arena_->pageAt(vid)
                 + static_cast<std::size_t>(i & mask_) * sizeof(int32_t);
        }

        /// @brief Compute the power-of-two slots-per-page split on first use
        ///        (the pool is not initialised at member-construction time).
        void ensureGeometry() {
            if (shift_ >= 0) return;
            const int32_t fit =
                arena_->pageBytes() / static_cast<int32_t>(sizeof(int32_t));
            assert(fit >= 1
                && "static_page_bytes too small for one hash-index slot");
            int32_t s = 0;
            while ((2 << s) <= fit) ++s;
            shift_ = s;
            mask_ = (1 << s) - 1;
            slotsPerPage_ = 1 << s;
        }

        LbArena* arena_;
        int32_t rootVid_ = kNoVid;       // empty / single data vid / directory vid
        int32_t capacity_ = 0;           // slot count (power of two when non-zero)
        int32_t numPages_ = 0;           // data pages held
        int32_t shift_ = -1;             // log2(slotsPerPage); -1 = not computed
        int32_t mask_ = 0;               // slotsPerPage - 1
        int32_t slotsPerPage_ = 0;       // 1 << shift_
    };

}
