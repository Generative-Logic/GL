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

#include "dirty_state.hpp"
#include "lb_arena.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace gl {

    /// @brief Contiguous-within-page replacement for `ArenaVector<T>` — the
    ///        statified container that wins back cache locality.
    ///
    /// @details
    /// Storage is a sequence of fixed-size pages taken from the owning
    /// `LbArena`'s page tier (`allocPage` / `freePage` / `pageAt`); elements
    /// pack TIGHTLY and CONTIGUOUSLY inside each page (`elemsPerPage` is the
    /// largest power of two that fits, so the within-page split is one shift +
    /// one mask). A sequential scan therefore walks contiguous memory and the
    /// hardware prefetcher engages — the property the scattered per-element
    /// `ArenaVector` lost.
    ///
    /// The page directory lives IN THE ARENA, never on the heap (the
    /// statification no-heap rule): the only heap-free per-container state is a
    /// handful of inline ints. The directory is adaptive, to keep the
    /// per-container page overhead low (`dirCap = pageBytes / sizeof(int32)`
    /// data-page vids per directory page — a power of two, e.g. 2048 at 8 KiB):
    ///   - EMPTY: `rootVid_ == kNoVid`, zero pages.
    ///   - ONE data page (`numPages_ == 1`): `rootVid_` IS that data page's vid
    ///     — no directory page at all.
    ///   - `2 .. dirCap` data pages: `rootVid_` is a single directory page
    ///     holding the data-page vids.
    ///   - `dirCap+1 .. dirCap²` data pages: `rootVid_` is a TWO-LEVEL root —
    ///     an L2 directory page of level-1 directory-page vids; each L1 page
    ///     holds up to `dirCap` data-page vids. When the single directory page
    ///     fills, it becomes L1 page 0 with NO copying. The page index splits
    ///     into (L1 index, slot) with one shift + one mask (`dirCap` is a power
    ///     of two), mirroring the within-page split. Overflowing `dirCap²`
    ///     asserts, naming the three-level directory that would follow — no real
    ///     per-LB container is expected to exceed it under LB-split.
    ///
    /// Pages are addressed by the arena's stable virtual ids (vids), resolved
    /// through `pageAt` at every access; nothing here caches a physical
    /// pointer, so the background compaction may rebind a vid to a different
    /// physical page between accesses without this container noticing
    /// (I-107).
    ///
    /// Deliberately an API SUBSET of `std::vector`, matched to `ArenaVector`'s
    /// public surface so a consumer's element type swaps with no other change:
    /// `push_back` / `size` / `empty` / `operator[]` / `addScalarToSuffix` /
    /// `erase(index)` /
    /// `clear` / `release` / copy-assign (cross-arena deep copy) / assign from
    /// `std::vector<T>` / `appendSpanBytes` / `bulkAppendBytes` (the deload
    /// element stream) / `liveBytes`. NO `data()`, NO iterators, NO `swap` —
    /// their absence forces the compiler to flag any consumer that still
    /// assumes contiguous whole-container storage.
    ///
    /// Geometry is computed lazily at first use, NOT in the constructor:
    /// `Memory` members construct before the JSON config parse initialises the
    /// pool, so the constructor must not read `pageBytes()`.
    ///
    /// @invariant `size()` elements live at logical indices `[0, size())` in
    ///            insertion order; element references stay valid while the LB
    ///            is resident and no preceding index is erased; they die at
    ///            `clear` / `release` / `erase` of an earlier index / deload /
    ///            compaction — never hold one across such a boundary.
    /// @see `LbArena` (the page source), `ArenaVector` (the scattered
    ///      predecessor it replaces).
    template <typename T>
    class PagedVector {
        static_assert(std::is_trivially_copyable<T>::value,
            "PagedVector elements must be trivially copyable - raw page bytes "
            "are streamed to SSD on deload");
        static_assert(alignof(T) <= 16,
            "pages are 16-byte aligned; over-aligned element types would read "
            "misaligned");

    public:
        /// @brief Element type, for generic consumers (the deload serializer's
        ///        container enumeration).
        using value_type = T;

        /// @brief Bind to the owning LB's arena and the aggregate's shared
        ///        dirty flag.
        ///
        /// @details
        /// Touches neither the arena nor `pageBytes()` (lazy geometry — see the
        /// class @details). Every mutator escalates `*dirty` so the deload path
        /// knows the on-disk image is stale; the mutation routes are CLOSED (no
        /// `data()`, no iterators), so the flag cannot be bypassed.
        ///
        /// @param arena The LB's page-dispensing arena. Must be non-null.
        /// @param dirty The aggregate's shared content-change state. Must be
        ///              non-null.
        explicit PagedVector(LbArena* arena, DirtyState* dirty)
            : arena_(arena), dirty_(dirty) {
            assert(arena_ != nullptr);
            assert(dirty_ != nullptr);
        }

        PagedVector(const PagedVector&) = delete;
        PagedVector(PagedVector&&) = delete;
        PagedVector& operator=(PagedVector&&) = delete;

        /// @brief Frees every held page back to the arena — with the
        ///        teardown-only residency branch for a RAW-deloaded LB.
        ///
        /// @details
        /// `Memory` declares the arena BEFORE its containers, so the arena is
        /// still alive when a container destructs and the pages can be returned.
        ///
        /// Teardown-only residency branch: when a RAW-deloaded `Memory` is
        /// destroyed, `Memory::releaseStaticBlocksRaw` returned every arena
        /// block WITHOUT walking the containers (the raw image preserves the
        /// vids, so a raw reload rebinds them — the near-memcpy win), so this
        /// container still holds a nonzero `rootVid_`/`numPages_` on a DELOADED
        /// arena. Its pages provably no longer exist (the blocks are back in
        /// the pool), so `freePage` must not run — reset the bookkeeping only.
        /// A DEFINED teardown state, NOT a swallowed failure (Rule 19): the
        /// branch lives ONLY in the destructor, so every LIVE `clear()` /
        /// `release()` on a cold arena still dies loudly on `freePage`'s
        /// residency assert — the guard cannot mask a mid-run bug. (A
        /// v3-deloaded container is already empty at destruction and takes the
        /// same branch harmlessly.)
        ~PagedVector() {
            if (!arena_->resident()) {
                rootVid_ = kNoVid;
                size_ = 0;
                numPages_ = 0;
                return;
            }
            clear();
        }

        /// @brief Number of elements.
        ///
        /// @details
        /// Asserts residency: a deloaded container reads as empty in RAM, and a
        /// silent size-0 answer would turn every loop over it into a no-op
        /// while the LB's RAM-side tables stay mutable — the pairing-corruption
        /// class the residency contract exists to stop
        /// (I-111).
        ///
        /// @return Logical element count.
        int32_t size() const {
            assert(arena_->resident()
                && "PagedVector::size() on a deloaded LB");
            return size_;
        }

        /// @brief Whether the container is empty.
        ///
        /// @return `true` when `size() == 0`. Asserts residency like `size()`.
        bool empty() const { return size() == 0; }

        /// @brief Data pages currently held (telemetry; discharge accounting).
        ///
        /// @return Number of data pages backing the elements.
        int32_t pagesHeld() const { return numPages_; }

        /// @brief Element access — read-only BY DESIGN (no mutable overload).
        ///
        /// @details
        /// One shift + one mask to split the index into (data page, slot), one
        /// directory lookup when there is more than one page, and one `pageAt`
        /// to resolve the data page. Asserts range and — via `pageAt` —
        /// residency. In-place writes are impossible through the public
        /// surface, which keeps the skip-unchanged deload contract airtight.
        ///
        /// @param i Logical index in `[0, size())`.
        /// @return Const reference, valid while the LB stays resident and no
        ///         preceding element is erased.
        GL_FORCEINLINE const T& operator[](int32_t i) const {
            assert(i >= 0 && i < size_);
            const int32_t dataVid = dataVidOf(i >> shift_);
            return *reinterpret_cast<const T*>(
                arena_->pageAt(dataVid)
                + static_cast<std::size_t>(i & mask_) * sizeof(T));
        }

        /// @brief In-place overwrite of element `i` — the one controlled mutable
        ///        door (`operator[]` stays read-only by design).
        ///
        /// @details
        /// `PagedVector` is otherwise const-access so the deload skip-unchanged
        /// contract cannot be bypassed silently. This single reviewed door
        /// (Rule 8/19) exists for the cold-map family's in-place value update
        /// (`ColdHashMap::setValueAt` — e.g. the `intKnownStatements` flag-bit OR)
        /// and the erase compaction. It escalates the dirty state to
        /// `Restructured`: an in-place write is NOT an append, so the next deload
        /// must be a full canonical rewrite, never a tail-delta — exactly the
        /// case the `DirtyState` doc names. Asserts range; residency is enforced
        /// by `writeSlot`/`pageAt` like every other mutator.
        ///
        /// @param i     Logical index in `[0, size())`.
        /// @param value Element to store.
        void setAt(int32_t i, const T& value) {
            assert(i >= 0 && i < size_);
            *dirty_ = DirtyState::Restructured;
            writeSlot(i, value);
        }

        /// @brief Add one scalar to every element in the suffix `[first,size)`
        ///        with one page-directory resolution per contiguous page run.
        ///
        /// @details
        /// CSR offset columns rebase a long suffix after an interior run splice.
        /// Calling `operator[]` plus `setAt` for every offset performs two virtual
        /// page-directory walks per integer. This door resolves each data page
        /// once, updates its contiguous elements in place, and escalates the dirty
        /// state once. The element order and resulting bytes are identical to the
        /// scalar loop. An empty suffix or zero delta is a defined no-op.
        ///
        /// @param first First element to change, in `[0,size()]`.
        /// @param delta Scalar added to each suffix element.
        /// @return Nothing.
        /// @invariant Every element before @p first is unchanged; every element at
        ///            or after it is increased by exactly @p delta; a non-empty,
        ///            non-zero update marks the aggregate `Restructured`.
        void addScalarToSuffix(int32_t first, T delta) {
            static_assert(std::is_arithmetic<T>::value,
                "PagedVector::addScalarToSuffix requires an arithmetic element");
            assert(first >= 0 && first <= size_);
            if (first == size_ || delta == static_cast<T>(0)) return;
            *dirty_ = DirtyState::Restructured;
            int32_t at = first;
            while (at < size_) {
                const int32_t within = at & mask_;
                const int32_t toPageEnd = (mask_ + 1) - within;
                const int32_t run = std::min(toPageEnd, size_ - at);
                T* p = reinterpret_cast<T*>(slotPtr(at));
                for (int32_t j = 0; j < run; ++j) p[j] += delta;
                at += run;
            }
        }

        /// @brief In-place overwrite of element `i` that does NOT escalate the
        ///        dirty state — the parallel-safe twin of `setAt` for
        ///        never-deloaded pooled containers.
        ///
        /// @details
        /// `setAt` escalates `*dirty_` to `Restructured` so the next deload is a
        /// full canonical rewrite. That shared-flag write makes `setAt` unsafe
        /// to call concurrently even on DISJOINT slots — two threads writing the
        /// one `*dirty_` byte is a data race regardless of the value. This twin
        /// writes only the element slot and leaves `*dirty_` untouched, so
        /// concurrent calls to DISJOINT indices are race-free (disjoint slots are
        /// disjoint bytes; `pageAt`/`writeSlot` only READ the frozen page table).
        ///
        /// The ONLY sanctioned use is a container on a NEVER-DELOADED pool whose
        /// dirty state is meaningless because it produces no deload image —
        /// specifically the pull-model mail cursor on the mail pool, advanced in
        /// the parallel phase-1 pull where each recipient writes only its own
        /// (disjoint) cursor cells ([`03_mail_system.md`]; I-94).
        /// On a DELOADABLE container it would silently skip the rewrite the change
        /// needs — a Rule-8/19 violation. Asserts range; residency is enforced by
        /// `writeSlot`/`pageAt` like every other accessor.
        ///
        /// @param i     Logical index in `[0, size())`.
        /// @param value Element to store.
        void setAtRelaxed(int32_t i, const T& value) {
            assert(i >= 0 && i < size_);
            writeSlot(i, value);
        }

        /// @brief Pointer to element `i` plus the number of elements contiguous
        ///        with it to the end of its page — the page-run primitive.
        ///
        /// @details
        /// Lets a caller walk a logical range as a sequence of contiguous spans
        /// (page-aware compare / hash / copy) and lets a reader detect a
        /// single-page range (when `runLen` reaches the range's length) for a
        /// zero-copy view. One shift + one mask + one resolve.
        ///
        /// @param i      Logical index in `[0, size())`.
        /// @param runLen [out] Elements contiguous from `i` — to the end of its
        ///               page or the logical end, whichever is nearer.
        /// @return Pointer to element `i`, readable for `runLen` elements.
        GL_FORCEINLINE const T* contiguousRun(int32_t i, int32_t& runLen) const {
            assert(i >= 0 && i < size_);
            const int32_t within = i & mask_;
            const int32_t toPageEnd = (mask_ + 1) - within;
            runLen = (toPageEnd < size_ - i) ? toPageEnd : (size_ - i);
            return reinterpret_cast<const T*>(
                arena_->pageAt(dataVidOf(i >> shift_))
                + static_cast<std::size_t>(within) * sizeof(T));
        }

        /// @brief Append one element, growing by one page on a boundary.
        ///
        /// @param value Element to copy in.
        void push_back(const T& value) {
            if (*dirty_ == DirtyState::Clean)
                *dirty_ = DirtyState::AppendedOnly;
            ensureGeometry();
            if (size_ == (numPages_ << shift_))
                appendDataVid(arena_->allocPage());
            writeSlot(size_, value);
            ++size_;
        }

        /// @brief Erase the element at index `i`, shifting successors left
        ///        (the `std::vector::erase` contract, page-aware).
        ///
        /// @details
        /// O(size - i) element copies across pages — fine for the cold
        /// maintenance paths that erase (`wipeSubtree`-style back-to-front
        /// loops). A tail page emptied by the shrink is freed eagerly.
        ///
        /// @param i Logical index in `[0, size())`.
        void erase(int32_t i) {
            assert(i >= 0 && i < size_);
            *dirty_ = DirtyState::Restructured;
            for (int32_t j = i; j + 1 < size_; ++j)
                writeSlot(j, (*this)[j + 1]);
            --size_;
            if (numPages_ > 0 && size_ <= ((numPages_ - 1) << shift_))
                freeLastDataPage();
        }

        /// @brief Insert `value` at index `i`, shifting elements `[i, size())`
        ///        one slot right (the `std::vector::insert` contract,
        ///        page-aware).
        ///
        /// @details
        /// The mirror of `erase(index)` — O(size - i) element copies across
        /// pages, growing by one page on a boundary. The cold-map family's
        /// set-valued store (`SetValueStore`) uses it to splice one value into
        /// the sorted middle of a key's CSR run, the operation `appendToTail`
        /// (tail only) cannot do. Marks the aggregate `Restructured` — an
        /// interior insert is not a tail-delta-eligible append. Single-threaded
        /// write side only (I-83).
        ///
        /// @param i     Logical index in `[0, size()]` (`i == size()` appends).
        /// @param value Element to copy in.
        void insertAt(int32_t i, const T& value) {
            assert(i >= 0 && i <= size_);
            *dirty_ = DirtyState::Restructured;
            ensureGeometry();
            if (size_ == (numPages_ << shift_))
                appendDataVid(arena_->allocPage());
            for (int32_t j = size_; j > i; --j)
                writeSlot(j, (*this)[j - 1]);
            writeSlot(i, value);
            ++size_;
        }

        /// @brief Replace the `oldLen` elements at `[pos, pos+oldLen)` with the
        ///        `newLen` elements at `src`, shifting the surviving tail exactly
        ///        ONCE — the variable-length range splice the blob value store's
        ///        whole-run replace (`HashMap::assignRun`) needs.
        ///
        /// @details
        /// The range generalization of `insertAt` (`oldLen == 0`) and a
        /// range-`erase` (`newLen == 0`): a grow extends the page run then shifts
        /// `[pos+oldLen, size())` right by `newLen-oldLen`; a shrink shifts that
        /// tail left then frees the emptied tail pages; finally `src` overwrites
        /// `[pos, pos+newLen)`. The shift and the fill are BULK page-span moves
        /// (`blockMove` / `writeBytesAt` — one `memmove` / `memcpy` per page span
        /// at memory bandwidth), NOT an element-by-element page resolve: ONE tail
        /// pass, O(|tail| + newLen) bytes touched with O((|tail|+newLen) / page)
        /// resolves, versus the O(changed * |tail|) a per-element
        /// `insertAt`/`erase` loop costs. Marks the aggregate
        /// `Restructured` (an interior resize is never a tail-delta-eligible
        /// append). `src` must not point into `*this` (the caller passes a heap
        /// buffer). Single-threaded write side only (I-83).
        ///
        /// @param pos    Range start in `[0, size()]`.
        /// @param oldLen Elements replaced; `pos + oldLen <= size()`.
        /// @param src    The `newLen` replacement elements; read only when
        ///               `newLen > 0`.
        /// @param newLen Replacement count; >= 0.
        void replaceRange(int32_t pos, int32_t oldLen, const T* src,
                          int32_t newLen) {
            assert(newLen >= 0 && (newLen == 0 || src != nullptr));
            replaceRangeGenerated(pos, oldLen, newLen, [&](const auto& sink) {
                if (newLen > 0) sink(src, newLen);
            });
        }

        /// @brief Replace one logical range from a replayable segmented source,
        ///        shifting the surviving tail exactly once.
        ///
        /// @details
        /// The generated-source twin of @ref replaceRange. It first resizes the
        /// destination gap with the same page-aware single `blockMove`, then
        /// invokes @p emit with a sink callable. The emitter feeds one or more
        /// contiguous `T` spans to that sink; together they must contain exactly
        /// @p newLen elements. Each span is copied directly into the already-sized
        /// destination through `writeBytesAt`, so a logical replacement larger
        /// than one arena block needs no contiguous staging allocation and still
        /// moves the old tail only once. The emitter is invoked exactly once.
        ///
        /// Single-threaded write side only (I-83). Marks the aggregate
        /// `Restructured`. Source spans must not alias `*this` across the resize;
        /// callers preserve any pool-backed source before entering.
        ///
        /// @tparam Emit Callable accepting one sink callable; the sink signature
        ///              is `void(const T* elements, int32_t count)`.
        /// @param pos    Range start in `[0, size()]`.
        /// @param oldLen Elements replaced; `pos + oldLen <= size()`.
        /// @param newLen Replacement element count; >= 0.
        /// @param emit   Replay-free producer that sends exactly @p newLen
        ///               elements to its supplied sink.
        /// @return Nothing.
        /// @invariant The logical prefix and suffix retain byte-identical order;
        ///            only the replacement range changes.
        /// @see replaceRange.
        template <typename Emit>
        void replaceRangeGenerated(int32_t pos, int32_t oldLen, int32_t newLen,
                                   Emit emit) {
            assert(pos >= 0 && oldLen >= 0 && pos + oldLen <= size_);
            assert(newLen >= 0);
            const int32_t delta = newLen - oldLen;
            if (delta > 0) {
                ensureGeometry();
                const int32_t oldSize = size_;
                for (int32_t k = 0; k < delta; ++k) {     // extend pages + size_
                    if (size_ == (numPages_ << shift_))
                        appendDataVid(arena_->allocPage());
                    ++size_;
                }
                blockMove(pos + oldLen, pos + oldLen + delta,
                          oldSize - (pos + oldLen));       // shift tail right
            } else if (delta < 0) {
                blockMove(pos + oldLen, pos + oldLen + delta,
                          size_ - (pos + oldLen));         // shift tail left
            }
            int32_t written = 0;
            const auto sink = [&](const T* elements, int32_t count) {
                assert(count >= 0 && written + count <= newLen);
                assert(count == 0 || elements != nullptr);
                if (count > 0) {
                    writeBytesAt(pos + written,
                        reinterpret_cast<const char*>(elements), count);
                    written += count;
                }
            };
            emit(sink);
            assert(written == newLen
                && "PagedVector::replaceRangeGenerated emitter length mismatch");
            if (delta < 0) {
                size_ += delta;
                while (numPages_ > 0 && size_ <= ((numPages_ - 1) << shift_))
                    freeLastDataPage();
            }
            *dirty_ = DirtyState::Restructured;
        }

        /// @brief Drop the tail down to `newSize` elements, freeing every page
        ///        that falls wholly past the new end.
        ///
        /// @details
        /// The bulk tail-removal the cold-map family's compacting `eraseIf` ends
        /// with: survivors are first copied to the front (via `setAt`), then the
        /// dead tail is dropped here in O(pages freed), not O(elements). Marks
        /// the aggregate `Restructured` — a shrink is not an append. Frees the
        /// emptied pages eagerly, demoting the directory shape as it shrinks (the
        /// `erase` page-reclaim, applied repeatedly); a partially-used final page
        /// is retained. `newSize == size()` is a no-op.
        ///
        /// @param newSize Retained element count, in `[0, size()]`.
        void truncate(int32_t newSize) {
            assert(newSize >= 0 && newSize <= size_);
            if (newSize == size_) return;
            *dirty_ = DirtyState::Restructured;
            size_ = newSize;
            while (numPages_ > 0 && size_ <= ((numPages_ - 1) << shift_))
                freeLastDataPage();
        }

        /// @brief Grow to `newSize` elements without initializing the new
        ///        slots — the reservation half of an in-place tail rewrite.
        ///
        /// @details
        /// Acquires the pages the new tail needs and advances the size; the
        /// caller fills every new slot before reading it (the in-place blob-run
        /// rebuild writes its whole tail backwards from the new end). Forces
        /// `Restructured`: a content-wise no-op here is still followed by the
        /// caller's rewrite of the moved tail. Single-threaded write side only
        /// (I-83).
        ///
        /// @param newSize The target element count; `>= size()`.
        /// @return Nothing.
        /// @invariant `size() == newSize` after return; every slot below the old
        ///            size keeps its content.
        /// @see truncate, blockMove, writeBytesAt.
        void growTo(int32_t newSize) {
            assert(newSize >= size_ && "PagedVector::growTo cannot shrink");
            if (newSize == size_) return;
            *dirty_ = DirtyState::Restructured;
            ensureGeometry();
            while (size_ < newSize) {
                if (size_ == (numPages_ << shift_))
                    appendDataVid(arena_->allocPage());
                const int32_t room = std::min(newSize - size_,
                                              (numPages_ << shift_) - size_);
                size_ += room;
            }
        }

        /// @brief Overwrite the existing slots `[at, at + n)` with `n`
        ///        elements from a foreign contiguous buffer — the typed public
        ///        face of the page-aware bulk fill.
        ///
        /// @details
        /// One `memcpy` per page span; the slots must already exist (a
        /// `growTo` precedes a tail fill) and @p src must not alias this
        /// vector's pages. Forces `Restructured` (an in-place rewrite).
        /// Single-threaded write side only (I-83).
        ///
        /// @param at  First destination slot; `at + n <= size()`.
        /// @param src The elements; read only when `n > 0`.
        /// @param n   Element count; `>= 0`.
        /// @return Nothing.
        /// @see growTo, blockMove, replaceRange.
        void writeRunAt(int32_t at, const T* src, int32_t n) {
            assert(n >= 0 && at >= 0 && at + n <= size_
                && "PagedVector::writeRunAt outside the existing slots");
            if (n == 0) return;
            *dirty_ = DirtyState::Restructured;
            writeBytesAt(at, reinterpret_cast<const char*>(src), n);
        }

        /// @brief Move the existing slots `[src, src + count)` to
        ///        `[dest, dest + count)` — the public face of the
        ///        overlap-safe page-aware `blockMove`.
        ///
        /// @details
        /// Both ranges must lie inside the existing slots (a `growTo` precedes
        /// an upward tail move); overlap in either direction is handled.
        /// Forces `Restructured`. Single-threaded write side only (I-83).
        ///
        /// @param src   First source slot.
        /// @param dest  First destination slot.
        /// @param count Elements; `<= 0` or `src == dest` is a no-op.
        /// @return Nothing.
        /// @see growTo, writeRunAt.
        void moveRange(int32_t src, int32_t dest, int32_t count) {
            if (count <= 0 || src == dest) return;
            assert(src >= 0 && dest >= 0 && src + count <= size_ && dest + count <= size_
                && "PagedVector::moveRange outside the existing slots");
            *dirty_ = DirtyState::Restructured;
            blockMove(src, dest, count);
        }

        /// @brief Drop all elements and free every page (data, level-1
        ///        directory, and L2 root, per the directory shape).
        ///
        /// @details
        /// A LIVE mutator — it requires a resident arena like every other page
        /// operation, and `freePage`'s own residency assert is the tripwire: a
        /// `clear` reaching a cold LB mid-run is a bug that must die loudly at
        /// its origin (Rule 19), never silently reset bookkeeping. The one
        /// lifecycle that legally meets a non-resident arena with live
        /// bookkeeping — destroying a RAW-deloaded `Memory` — is handled by the
        /// residency branch in `~PagedVector`, never here.
        void clear() {
            *dirty_ = DirtyState::Restructured;
            const int32_t dirCap = dirMask_ + 1;
            if (numPages_ == 1) {
                arena_->freePage(rootVid_);
            } else if (numPages_ > 1 && numPages_ <= dirCap) {   // single directory page
                const int32_t* dir = reinterpret_cast<const int32_t*>(
                    arena_->pageAt(rootVid_));
                for (int32_t p = 0; p < numPages_; ++p)
                    arena_->freePage(dir[p]);
                arena_->freePage(rootVid_);   // the directory page itself
            } else if (numPages_ > dirCap) {                     // two-level
                const int32_t* l2 = reinterpret_cast<const int32_t*>(
                    arena_->pageAt(rootVid_));
                const int32_t l1Count = (numPages_ + dirMask_) >> dirShift_;
                for (int32_t li = 0; li < l1Count; ++li) {
                    const int32_t* l1 = reinterpret_cast<const int32_t*>(
                        arena_->pageAt(l2[li]));
                    const int32_t base = li << dirShift_;
                    const int32_t inThis = std::min(dirCap, numPages_ - base);
                    for (int32_t s = 0; s < inThis; ++s)
                        arena_->freePage(l1[s]);
                    arena_->freePage(l2[li]);   // the level-1 directory page
                }
                arena_->freePage(rootVid_);     // the L2 root
            }
            rootVid_ = kNoVid;
            size_ = 0;
            numPages_ = 0;
        }

        /// @brief Identical to `clear` for the paged layout — there is no heap
        ///        capacity to drop. Kept for `ArenaVector` API parity.
        void release() { clear(); }

        /// @brief Deep copy (element-wise), including across two different LB
        ///        arenas — the LB-clone path.
        ///
        /// @param other Source vector (unchanged).
        /// @return `*this`.
        PagedVector& operator=(const PagedVector& other) {
            if (this == &other) return *this;
            clear();
            for (int32_t i = 0; i < other.size_; ++i) push_back(other[i]);
            return *this;
        }

        /// @brief Rebuild from a `std::vector<T>` — the scope-teardown
        ///        filter-and-reassign sites.
        ///
        /// @param src Source elements, copied in order.
        /// @return `*this`.
        PagedVector& operator=(const std::vector<T>& src) {
            clear();
            for (const T& value : src) push_back(value);
            return *this;
        }

        /// @brief Append the tightly-packed bytes of rows `[fromRow, size())`
        ///        to a byte buffer — the canonical deload stream (one `memcpy`
        ///        per page span).
        ///
        /// @details
        /// Elements are tightly packed within a page, so the concatenation of
        /// each page's used span is byte-identical to a per-element walk — and
        /// byte-identical to `ArenaVector::appendSpanBytes` for the same
        /// content, so swapping a container's type does not change its deload
        /// image (I-103). Read-only.
        ///
        /// @param out     Destination byte buffer (appended to).
        /// @param fromRow First row to emit, in `[0, size()]`.
        void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
            assert(fromRow >= 0 && fromRow <= size_);
            int32_t row = fromRow;
            while (row < size_) {
                const int32_t slot = row & mask_;
                const int32_t epp = mask_ + 1;
                const int32_t rows = std::min(epp - slot, size_ - row);
                const char* span =
                    arena_->pageAt(dataVidOf(row >> shift_))
                    + static_cast<std::size_t>(slot) * sizeof(T);
                const std::size_t bytes =
                    static_cast<std::size_t>(rows) * sizeof(T);
                const std::size_t base = out.size();
                out.resize(base + bytes);
                std::memcpy(out.data() + base, span, bytes);
                row += rows;
            }
        }

        /// @brief Append `rowCount` elements from a tightly-packed byte stream
        ///        — the bulk reload path (one `memcpy` per page span).
        ///
        /// @details
        /// A mutator like any other (escalates the dirty state to
        /// `Restructured`; `ensureLoaded` resets it after the reload). The
        /// deload machinery is the intended caller; production flow keeps using
        /// `push_back`.
        ///
        /// @param bytes    Source stream of `rowCount * sizeof(T)` bytes.
        /// @param rowCount Elements to append; >= 0.
        void bulkAppendBytes(const char* bytes, int64_t rowCount) {
            assert(rowCount >= 0);
            *dirty_ = DirtyState::Restructured;
            ensureGeometry();
            fillRun(bytes, rowCount);
        }

        /// @brief Bulk-append `n` elements with APPEND-only dirty semantics.
        ///
        /// @details
        /// The same page-span fill as `bulkAppendBytes`, but it escalates the
        /// dirty state only `Clean -> AppendedOnly` (never `Restructured`) — an
        /// append keeps the deload tail-delta eligible, whereas
        /// `bulkAppendBytes` is the reload path and marks a full restructure.
        /// The cold string interner's content append uses this.
        ///
        /// @param src Source elements (`n * sizeof(T)` bytes).
        /// @param n   Element count; >= 0.
        void appendRun(const T* src, int32_t n) {
            assert(n >= 0);
            if (n == 0) return;
            if (*dirty_ == DirtyState::Clean)
                *dirty_ = DirtyState::AppendedOnly;
            ensureGeometry();
            fillRun(reinterpret_cast<const char*>(src),
                    static_cast<int64_t>(n));
        }

        /// @brief Append `n` elements as ONE within-page run — padding to the
        ///        next page first if they would straddle — and return the run's
        ///        start index. The cold-string no-straddle interning primitive.
        ///
        /// @details
        /// A cold string must read back as one contiguous span (`view`), so its
        /// bytes may not cross a page boundary. When the run would not fit the
        /// current page's remainder, that tail is skipped (filled with zero
        /// filler) and the run lands wholly inside a fresh page; the start index
        /// returned points PAST the padding. With the run thus guaranteed
        /// page-resident, the fill delegates to `appendRun` (same APPENDED-only
        /// dirty semantics). The run must fit one page (`n <= elemsPerPage`); a
        /// larger one asserts — raise `static_page_bytes`.
        ///
        /// The padding makes the page byte stream NON-dense, so a consumer that
        /// streams this container for deload must walk it per logical element
        /// (the `ColdStringTable` content dump does) — NOT via `appendSpanBytes`,
        /// which would emit the filler. No other paged container pads, so their
        /// `appendSpanBytes` stays the byte-identical dense stream.
        ///
        /// @param src Source elements (`n * sizeof(T)` bytes); read only when
        ///            `n > 0`.
        /// @param n   Element count; `0 <= n <= elemsPerPage`.
        /// @return The logical index where the run starts (after any padding);
        ///         `size()` when `n == 0`.
        int32_t appendRunNoStraddle(const T* src, int32_t n) {
            assert(n >= 0);
            ensureGeometry();
            const int32_t epp = mask_ + 1;
            assert(n <= epp
                && "PagedVector::appendRunNoStraddle run exceeds one page — "
                   "raise static_page_bytes or shorten the element run");
            if (n == 0) return size_;
            const int32_t within = size_ & mask_;
            if (within != 0 && within + n > epp) {
                // Skip the current page's tail so the run starts fresh; the
                // filler is never read (the index records the post-pad start)
                // nor deloaded (the content dump walks per element).
                const int32_t pad = epp - within;
                std::memset(
                    arena_->pageAt(dataVidOf(size_ >> shift_))
                        + static_cast<std::size_t>(within) * sizeof(T),
                    0, static_cast<std::size_t>(pad) * sizeof(T));
                if (*dirty_ == DirtyState::Clean)
                    *dirty_ = DirtyState::AppendedOnly;
                size_ += pad;
            }
            const int32_t start = size_;
            appendRun(src, n);
            return start;
        }

        /// @brief Live page footprint in bytes (the steward's pressure input).
        ///
        /// @details
        /// Counts the held pages — the data pages plus the directory pages: one
        /// directory page in the single-directory shape, or `ceil(numPages_ /
        /// dirCap)` level-1 pages plus one L2 root in the two-level shape —
        /// times `pageBytes`. Unlike `ArenaVector::liveBytes` (a logical element
        /// measure) this is the PHYSICAL pages the container pins, which is what
        /// the page-tier reclamation reasons about.
        ///
        /// @return Held-page bytes; 0 when empty.
        int64_t liveBytes() const {
            if (numPages_ == 0) return 0;
            const int32_t dirCap = dirMask_ + 1;
            int64_t held = numPages_;                       // data pages
            if (numPages_ > 1 && numPages_ <= dirCap) {
                held += 1;                                  // one directory page
            } else if (numPages_ > dirCap) {
                const int64_t l1Count =
                    (static_cast<int64_t>(numPages_) + dirMask_) >> dirShift_;
                held += l1Count + 1;                        // L1 pages + L2 root
            }
            return held * static_cast<int64_t>(arena_->pageBytes());
        }

    private:
        /// @brief Sentinel "no vid" for the empty / unbound directory root.
        static constexpr int32_t kNoVid = -1;

        /// @brief The vid of data page `pageIdx` (`0 <= pageIdx < numPages_`).
        ///
        /// @details
        /// Three shapes (see the class @details directory ladder):
        ///   - one data page (`numPages_ == 1`): its vid IS `rootVid_`, no
        ///     directory page exists;
        ///   - single directory page (`numPages_ <= dirCap`): `rootVid_` is the
        ///     directory page and the vid is its `pageIdx`-th entry;
        ///   - two-level (`numPages_ > dirCap`): `rootVid_` is the L2 root; the
        ///     page index splits into the L1-page index (`pageIdx >> dirShift_`)
        ///     and the within-L1 slot (`pageIdx & dirMask_`), one extra `pageAt`
        ///     resolve.
        ///
        /// @param pageIdx Data-page index.
        /// @return The data page's vid.
        GL_FORCEINLINE int32_t dataVidOf(int32_t pageIdx) const {
            if (numPages_ == 1) return rootVid_;
            const int32_t* root = reinterpret_cast<const int32_t*>(
                arena_->pageAt(rootVid_));
            if (numPages_ <= dirMask_ + 1) return root[pageIdx];
            const int32_t l1vid = root[pageIdx >> dirShift_];
            return reinterpret_cast<const int32_t*>(
                arena_->pageAt(l1vid))[pageIdx & dirMask_];
        }

        /// @brief In-place slot write — private: the only callers are the
        ///        mutators above, which own the dirty marking.
        ///
        /// @param i     Slot in `[0, size_]` (one past the end on append).
        /// @param value Element to store.
        GL_FORCEINLINE void writeSlot(int32_t i, const T& value) {
            const int32_t dataVid = dataVidOf(i >> shift_);
            *reinterpret_cast<T*>(arena_->pageAt(dataVid)
                + static_cast<std::size_t>(i & mask_) * sizeof(T)) = value;
        }

        /// @brief Append `count` elements from raw bytes across page spans —
        ///        the shared fill core of `bulkAppendBytes` and `appendRun`.
        ///
        /// @details
        /// Does NOT touch the dirty state (each public caller sets its own,
        /// since reload and append differ). One `memcpy` per page span.
        ///
        /// @param bytes Source bytes (`count * sizeof(T)`).
        /// @param count Element count; >= 0.
        void fillRun(const char* bytes, int64_t count) {
            int64_t done = 0;
            while (done < count) {
                if (size_ == (numPages_ << shift_))
                    appendDataVid(arena_->allocPage());
                const int32_t slot = size_ & mask_;
                const int32_t epp = mask_ + 1;
                const int32_t rows = static_cast<int32_t>(
                    std::min<int64_t>(epp - slot, count - done));
                std::memcpy(
                    arena_->pageAt(dataVidOf(size_ >> shift_))
                        + static_cast<std::size_t>(slot) * sizeof(T),
                    bytes + static_cast<std::size_t>(done) * sizeof(T),
                    static_cast<std::size_t>(rows) * sizeof(T));
                size_ += rows;
                done += rows;
            }
        }

        /// @brief Writable byte pointer to slot `i` — the page resolve shared by
        ///        the bulk block-move and fill.
        ///
        /// @details
        /// One shift + one mask + one `dataVidOf` resolve, mirroring `writeSlot`
        /// without the per-element store. The slot's page must already exist.
        ///
        /// @param i Slot whose page is allocated.
        /// @return Writable pointer to element `i` (valid for one element).
        GL_FORCEINLINE char* slotPtr(int32_t i) {
            return arena_->pageAt(dataVidOf(i >> shift_))
                + static_cast<std::size_t>(i & mask_) * sizeof(T);
        }

        /// @brief Page-aware block move of `count` elements `[src, src+count)` ->
        ///        `[dest, dest+count)` — one `memmove` per page-bounded span.
        ///
        /// @details
        /// The bulk replacement for `replaceRange`'s former element-by-element
        /// shift (an `operator[]` + `writeSlot` page resolve PER element —
        /// pathological for a char pool, where it relocated one BYTE per resolve
        /// and dominated the cold-blob-map insert). Copies in the overlap-safe
        /// direction — backward when `dest > src`, forward otherwise — and bounds
        /// every span by BOTH operands' distance to their page edge, so each
        /// `memmove` stays inside one source page and one destination page (which
        /// may be the same page, hence `memmove` not `memcpy`). All touched slots
        /// must already exist (a grow precedes a right shift). O(count) bytes at
        /// memory bandwidth, O(count / page) resolves. Single-threaded write side
        /// only (I-83).
        ///
        /// @param src   First source slot.
        /// @param dest  First destination slot.
        /// @param count Elements to move; `count <= 0` or `src == dest` is a
        ///              no-op.
        void blockMove(int32_t src, int32_t dest, int32_t count) {
            if (count <= 0 || src == dest) return;
            const int32_t epp = mask_ + 1;
            if (dest > src) {                          // overlap-safe: high -> low
                int32_t s = src + count, d = dest + count, rem = count;
                while (rem > 0) {
                    const int32_t chunk = std::min(rem,
                        std::min(((s - 1) & mask_) + 1, ((d - 1) & mask_) + 1));
                    std::memmove(slotPtr(d - chunk), slotPtr(s - chunk),
                                 static_cast<std::size_t>(chunk) * sizeof(T));
                    s -= chunk; d -= chunk; rem -= chunk;
                }
            } else {                                   // overlap-safe: low -> high
                int32_t s = src, d = dest, rem = count;
                while (rem > 0) {
                    const int32_t chunk = std::min(rem,
                        std::min(epp - (s & mask_), epp - (d & mask_)));
                    std::memmove(slotPtr(d), slotPtr(s),
                                 static_cast<std::size_t>(chunk) * sizeof(T));
                    s += chunk; d += chunk; rem -= chunk;
                }
            }
        }

        /// @brief Page-aware bulk write of `count` elements from `bytes` into the
        ///        existing slots `[at, at+count)` — the fill half of
        ///        `replaceRange`.
        ///
        /// @details
        /// Replaces the former per-element `writeSlot(pos+j, src[j])` loop with
        /// one `memcpy` per page span. The slots must already exist; `bytes` is a
        /// contiguous heap buffer that must not alias `*this`.
        ///
        /// @param at    First destination slot.
        /// @param bytes Source bytes (`count * sizeof(T)`); read only when
        ///              `count > 0`.
        /// @param count Element count; >= 0.
        void writeBytesAt(int32_t at, const char* bytes, int32_t count) {
            const int32_t epp = mask_ + 1;
            int32_t d = at, done = 0;
            while (done < count) {
                const int32_t chunk = std::min(count - done, epp - (d & mask_));
                std::memcpy(slotPtr(d),
                            bytes + static_cast<std::size_t>(done) * sizeof(T),
                            static_cast<std::size_t>(chunk) * sizeof(T));
                d += chunk; done += chunk;
            }
        }

        /// @brief Record a freshly allocated data-page vid, promoting the
        ///        directory shape (empty -> inline -> single directory page ->
        ///        two-level directory) as needed.
        ///
        /// @details
        /// The `numPages_ == dirCap` case promotes the full single directory
        /// page to two-level by making it L1 page 0 of a fresh L2 root (no entry
        /// copy) and starting a second L1 page for `vid`. Past two levels
        /// (`dirCap²` data pages) the three-level `assert` fires — the tripwire
        /// that the one-level overflow assert used to be, moved up one level.
        ///
        /// @param vid The new data page's vid.
        void appendDataVid(int32_t vid) {
            if (numPages_ == 0) {                 // empty -> single inline vid
                rootVid_ = vid;
                numPages_ = 1;
                return;
            }
            if (numPages_ == 1) {                 // inline -> single directory page
                const int32_t firstData = rootVid_;
                const int32_t dirVid = arena_->allocPage();
                int32_t* dir = reinterpret_cast<int32_t*>(
                    arena_->pageAt(dirVid));
                dir[0] = firstData;
                dir[1] = vid;
                rootVid_ = dirVid;
                numPages_ = 2;
                return;
            }
            const int32_t dirCap = dirMask_ + 1;
            if (numPages_ < dirCap) {             // single directory page, room
                reinterpret_cast<int32_t*>(
                    arena_->pageAt(rootVid_))[numPages_] = vid;
                ++numPages_;
                return;
            }
            if (numPages_ == dirCap) {            // single directory FULL -> two-level
                const int32_t oldDir = rootVid_;  // becomes L1 page 0 (already full)
                const int32_t l2vid = arena_->allocPage();
                const int32_t newL1 = arena_->allocPage();
                int32_t* l2 = reinterpret_cast<int32_t*>(arena_->pageAt(l2vid));
                l2[0] = oldDir;
                l2[1] = newL1;
                reinterpret_cast<int32_t*>(arena_->pageAt(newL1))[0] = vid;
                rootVid_ = l2vid;
                ++numPages_;                      // numPages_ = dirCap + 1
                arena_->recordTwoLevelPromotion(numPages_);   // telemetry only
                return;
            }
            assert(static_cast<int64_t>(numPages_)
                       < static_cast<int64_t>(dirCap) * dirCap
                && "PagedVector exceeded the two-level directory — add the "
                   "three-level directory for containers past this size");
            int32_t* l2 = reinterpret_cast<int32_t*>(arena_->pageAt(rootVid_));
            const int32_t l1Index = numPages_ >> dirShift_;
            const int32_t slot = numPages_ & dirMask_;
            if (slot == 0) {                      // start a fresh L1 page
                const int32_t newL1 = arena_->allocPage();
                l2[l1Index] = newL1;
                reinterpret_cast<int32_t*>(arena_->pageAt(newL1))[0] = vid;
            } else {                              // room in the current L1 page
                reinterpret_cast<int32_t*>(
                    arena_->pageAt(l2[l1Index]))[slot] = vid;
            }
            ++numPages_;
        }

        /// @brief Free the last data page, demoting the directory shape
        ///        (two-level -> single directory page -> inline -> empty) as it
        ///        shrinks.
        ///
        /// @details
        /// The mirror of `appendDataVid`. A two-level container whose last L1
        /// page empties frees that L1 page; at the `dirCap+1 -> dirCap` boundary
        /// it collapses back to a single directory page (L1 page 0, already
        /// holding exactly `dirCap` entries, becomes the directory page; the L2
        /// root and the emptied last L1 page are freed).
        void freeLastDataPage() {
            arena_->freePage(dataVidOf(numPages_ - 1));
            const int32_t dirCap = dirMask_ + 1;
            if (numPages_ == 1) {                 // single page -> empty
                rootVid_ = kNoVid;
                numPages_ = 0;
                return;
            }
            if (numPages_ == 2) {                 // directory -> inline
                const int32_t firstData = reinterpret_cast<const int32_t*>(
                    arena_->pageAt(rootVid_))[0];
                arena_->freePage(rootVid_);       // free the directory page
                rootVid_ = firstData;
                numPages_ = 1;
                return;
            }
            if (numPages_ <= dirCap) {            // stays a single directory page
                --numPages_;
                return;
            }
            if (numPages_ == dirCap + 1) {        // two-level -> single directory
                int32_t* l2 = reinterpret_cast<int32_t*>(arena_->pageAt(rootVid_));
                const int32_t firstL1 = l2[0];    // the full directory page (dirCap entries)
                const int32_t lastL1 = l2[1];     // held the just-freed entry
                arena_->freePage(lastL1);
                arena_->freePage(rootVid_);       // free the L2 root
                rootVid_ = firstL1;
                numPages_ = dirCap;
                return;
            }
            const int32_t freedIdx = numPages_ - 1;
            if ((freedIdx & dirMask_) == 0) {     // freed slot 0 -> its L1 page is now empty
                int32_t* l2 = reinterpret_cast<int32_t*>(arena_->pageAt(rootVid_));
                arena_->freePage(l2[freedIdx >> dirShift_]);
            }
            --numPages_;
        }

        /// @brief Compute the power-of-two elements-per-page split AND the
        ///        directory fan-out on first use (the pool is not initialised at
        ///        member-construction time).
        ///
        /// @details
        /// `dirCap = pageBytes / sizeof(int32)` is the data-page-vid fan-out of
        /// one directory page; it is already a power of two (the page size is),
        /// so the two-level page-index split is `dirShift_` / `dirMask_` —
        /// exact, one shift + one mask like the within-page element split.
        void ensureGeometry() {
            if (shift_ >= 0) return;
            const int32_t fit =
                arena_->pageBytes() / static_cast<int32_t>(sizeof(T));
            assert(fit >= 1
                && "static_page_bytes too small for one element of T");
            int32_t s = 0;
            while ((2 << s) <= fit) ++s;
            shift_ = s;
            mask_ = (1 << s) - 1;
            const int32_t dirFit =
                arena_->pageBytes() / static_cast<int32_t>(sizeof(int32_t));
            assert(dirFit >= 2
                && "static_page_bytes too small for a 2-entry directory page");
            int32_t ds = 0;
            while ((2 << ds) <= dirFit) ++ds;
            dirShift_ = ds;
            dirMask_ = (1 << ds) - 1;
        }

        LbArena* arena_;
        DirtyState* dirty_;          // aggregate's content-change state
        int32_t rootVid_ = kNoVid;   // empty / single data vid / directory vid / L2 root vid
        int32_t size_ = 0;           // logical element count
        int32_t numPages_ = 0;       // data pages held
        int32_t shift_ = -1;         // log2(elemsPerPage); -1 = not computed
        int32_t mask_ = 0;           // elemsPerPage - 1
        int32_t dirShift_ = 0;       // log2(dirCap); directory-page-index split
        int32_t dirMask_ = 0;        // dirCap - 1
    };

}
