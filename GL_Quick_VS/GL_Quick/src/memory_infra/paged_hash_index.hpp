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
    /// Beyond `slotsPerPage²` data pages, an L3 root holds L2 roots with the
    /// same L1 shape. Smaller arrays retain their original directory depth.
    /// Coverage beyond three directory levels asserts; the signed 32-bit slot
    /// count remains an independent limit, with widened page-count arithmetic.
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

        /// @brief Frees every held page back to the arena — with the
        ///        teardown-only residency branch for a RAW-deloaded LB.
        ///
        /// @details
        /// Teardown-only residency branch, the twin of `~PagedVector`'s: a
        /// RAW-deloaded LB's throw-away bucket index (`ColdHashSet::buckets_`)
        /// still holds `rootVid_`/`numPages_` on a DELOADED arena, because
        /// `Memory::releaseStaticBlocksRaw` skipped the container walk (the raw
        /// image carries the bucket pages verbatim). Its pages no longer exist
        /// (the blocks are back in the pool), so `freePage` must not run —
        /// reset the bookkeeping only. A DEFINED teardown state (Rule 19), not
        /// a swallowed failure: the branch lives ONLY in the destructor, so a
        /// LIVE `clear()` / `reset()` on a cold arena still dies loudly on
        /// `freePage`'s residency assert.
        ~PagedHashIndex() {
            if (capacity_ != 0 && !arena_->resident()) {
                rootVid_ = kNoVid;
                capacity_ = 0;
                numPages_ = 0;
                return;
            }
            clear();
        }

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
        /// directory page, two-level or three-level directory. An L3 root
        /// points at L2 roots, each covering `slotsPerPage²` data pages.
        /// Coverage is checked before filling the directory. The page count
        /// uses widened arithmetic even when `cap` is near its int32 limit.
        ///
        /// @param cap Slot capacity to allocate; `>= 0`.
        void reset(int32_t cap) {
            clear();
            assert(cap >= 0);
            if (cap == 0) return;
            ensureGeometry();
            capacity_ = cap;
            numPages_ = static_cast<int32_t>((static_cast<int64_t>(cap) + slotsPerPage_ - 1) >> shift_);
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
            } else {
                const int64_t span = static_cast<int64_t>(dirCap) * dirCap;
                assert(static_cast<int64_t>(numPages_) <= span * dirCap
                    && "PagedHashIndex exceeded the three-level directory");
                const auto fillL2 = [&](int32_t root, int32_t pages) {
                    int32_t* l2 = reinterpret_cast<int32_t*>(arena_->pageAt(root));
                    for (int64_t first = 0; first < pages; first += dirCap) {
                        const int32_t l1vid = arena_->allocPage();
                        l2[first >> shift_] = l1vid;
                        int32_t* l1 = reinterpret_cast<int32_t*>(arena_->pageAt(l1vid));
                        for (int32_t slot = 0; slot < std::min<int64_t>(dirCap, pages - first); ++slot) {
                            const int32_t vid = arena_->allocPage();
                            l1[slot] = vid;
                            std::memset(arena_->pageAt(vid), 0, pageBytes);
                        }
                    }
                };
                rootVid_ = arena_->allocPage();
                if (numPages_ <= span) {
                    fillL2(rootVid_, numPages_);
                } else {
                    int32_t* l3 = reinterpret_cast<int32_t*>(arena_->pageAt(rootVid_));
                    for (int64_t first = 0; first < numPages_; first += span) {
                        const int32_t l2vid = arena_->allocPage();
                        l3[first / span] = l2vid;
                        fillL2(l2vid, static_cast<int32_t>(std::min<int64_t>(span, numPages_ - first)));
                    }
                }
                arena_->recordTwoLevelPromotion(numPages_);   // telemetry only
            }
        }

        /// @brief Free every held page (data and all occupied directory levels,
        ///        per the directory shape) and reset to empty.
        ///
        /// @details
        /// A LIVE operation — it requires a resident arena, and `freePage`'s
        /// own residency assert is the tripwire: a `clear` reaching a cold LB
        /// mid-run is a bug that must die loudly at its origin (Rule 19), never
        /// silently reset bookkeeping. The one lifecycle that legally meets a
        /// non-resident arena with live bookkeeping — destroying a RAW-deloaded
        /// `Memory` — is handled by the residency branch in `~PagedHashIndex`,
        /// never here.
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
            } else {
                const int64_t span = static_cast<int64_t>(dirCap) * dirCap;
                const auto freeL2 = [&](int32_t root, int32_t pages) {
                    const int32_t* l2 = reinterpret_cast<const int32_t*>(arena_->pageAt(root));
                    for (int64_t first = 0; first < pages; first += dirCap) {
                        const int32_t l1vid = l2[first >> shift_];
                        const int32_t* l1 = reinterpret_cast<const int32_t*>(arena_->pageAt(l1vid));
                        for (int32_t slot = 0; slot < std::min<int64_t>(dirCap, pages - first); ++slot)
                            arena_->freePage(l1[slot]);
                        arena_->freePage(l1vid);
                    }
                    arena_->freePage(root);
                };
                if (numPages_ <= span) {
                    freeL2(rootVid_, numPages_);
                } else {
                    const int32_t* l3 = reinterpret_cast<const int32_t*>(arena_->pageAt(rootVid_));
                    for (int64_t first = 0; first < numPages_; first += span)
                        freeL2(l3[first / span], static_cast<int32_t>(std::min<int64_t>(span, numPages_ - first)));
                    arena_->freePage(rootVid_);
                }
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
                held += l1Count + 1;                        // L1 pages + root
                const int64_t span = static_cast<int64_t>(dirCap) * dirCap;
                if (numPages_ > span)
                    held += (static_cast<int64_t>(numPages_) + span - 1) / span; // L2 pages
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
        GL_FORCEINLINE char* slotPtr(int32_t i) {
            const int32_t pageIdx = i >> shift_;
            int32_t vid;
            if (numPages_ == 1) {
                vid = rootVid_;
            } else if (numPages_ <= mask_ + 1) {        // single directory page
                vid = reinterpret_cast<const int32_t*>(
                          arena_->pageAt(rootVid_))[pageIdx];
            } else {
                const int32_t* root = reinterpret_cast<const int32_t*>(arena_->pageAt(rootVid_));
                if (numPages_ > static_cast<int64_t>(slotsPerPage_) * slotsPerPage_)
                    root = reinterpret_cast<const int32_t*>(arena_->pageAt(
                        root[static_cast<int64_t>(pageIdx) >> (2 * shift_)]));
                const int32_t l1vid = root[(pageIdx >> shift_) & mask_];
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
        GL_FORCEINLINE const char* slotPtr(int32_t i) const {
            const int32_t pageIdx = i >> shift_;
            int32_t vid;
            if (numPages_ == 1) {
                vid = rootVid_;
            } else if (numPages_ <= mask_ + 1) {        // single directory page
                vid = reinterpret_cast<const int32_t*>(
                          arena_->pageAt(rootVid_))[pageIdx];
            } else {
                const int32_t* root = reinterpret_cast<const int32_t*>(arena_->pageAt(rootVid_));
                if (numPages_ > static_cast<int64_t>(slotsPerPage_) * slotsPerPage_)
                    root = reinterpret_cast<const int32_t*>(arena_->pageAt(
                        root[static_cast<int64_t>(pageIdx) >> (2 * shift_)]));
                const int32_t l1vid = root[(pageIdx >> shift_) & mask_];
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
