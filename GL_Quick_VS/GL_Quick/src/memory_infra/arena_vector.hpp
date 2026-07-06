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

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>
#include <vector>

namespace gl {

    /// @brief Arena-backed replacement for `std::vector<T>` — scattered
    ///        elements addressed through a two-level virtual-offset index.
    ///
    /// @details
    /// Every element is its own
    /// small `LbArena` allocation, so a container's elements are scattered
    /// through the LB's one arena, interleaved with every other container's
    /// (the user's "distributed over the block with interruptions" model).
    /// The **spine** indexes them and is two-level so it holds arbitrarily
    /// many elements despite a single allocation being capped at one block:
    ///
    /// - **chunks** — fixed runs of `kSpineChunk` (1024) `ArenaOffset`s
    ///   (4 KiB each); `chunk[within]` is the virtual offset of an element.
    /// - **directory** — a run of chunk offsets (`dir[chunk]` resolves to a
    ///   chunk), grown by reallocation (old run becomes a hole the compaction
    ///   reclaims).
    ///
    /// `operator[](i)` splits `i` into `(chunk = i >> 10, within = i & 1023)`,
    /// resolves the directory to the chunk and the chunk to the element — two
    /// index resolves plus the element resolve, all O(1). Both the spine and
    /// the elements are virtual offsets, so the whole structure is
    /// position-independent: it deloads and reloads onto fresh physical
    /// blocks unchanged. The directory itself is one allocation (≤ one block,
    /// so ≤ 65536 chunks ⇒ ≤ ~67M elements; beyond that `LbArena::alloc`
    /// asserts — far past any real LB).
    ///
    /// Deliberate API SUBSET of `std::vector`: `push_back` / `size` / `empty`
    /// / `operator[]` / `erase(index)` / `clear` / `release` / copy-assign
    /// (cross-arena deep copy for LB clones) / assign from `std::vector<T>`
    /// (scope-teardown rebuilds) / `appendSpanBytes` / `bulkAppendBytes`
    /// (the canonical element-stream deload). NO `data()`, NO iterators, NO
    /// `swap` — their absence forces the compiler to enumerate every consumer
    /// still assuming contiguous storage.
    ///
    /// Element references stay valid across growth while the LB is resident:
    /// an element never moves (the arena only bumps forward), and a directory
    /// growth copies the chunk-offset values verbatim, so `&operator[](i)`
    /// resolves to the same element. They die at `clear` / `release` / a
    /// preceding `erase` (the spine shifts) and at the copying compaction
    /// (offsets are reassigned). Per I-107:
    /// never hold one across a deload- or compaction-capable boundary.
    ///
    /// `operator[]` resolves through the spine on every call; the burst hot
    /// path takes a transient resolved-pointer view instead (built once at
    /// phase-2 entry — `IntStmtView`).
    ///
    /// @invariant `size()` elements live at logical indices `[0, size())` in
    ///            insertion order; storage freed by `erase` / `clear` /
    ///            `release` becomes a hole the compaction reclaims (the bump
    ///            arena frees only its tail).
    /// @see `LbArena` (offset source), `IntStmtView` (the burst-time resolved
    ///      view over the four statement vectors built on this).
    template <typename T>
    class ArenaVector {
        static_assert(std::is_trivially_copyable<T>::value,
            "ArenaVector elements must be trivially copyable - raw element "
            "bytes are streamed to SSD on deload");
        static_assert(alignof(T) <= 16,
            "arena allocations are at most 16-byte aligned; over-aligned "
            "element types would read misaligned");

    public:
        /// @brief Element type, for generic consumers (the deload
        ///        serializer's container enumeration).
        using value_type = T;

        /// @brief Bind to the owning LB's arena and the aggregate's shared
        ///        dirty flag.
        ///
        /// @details
        /// Does not touch the arena (no allocation) — geometry is lazy in
        /// `LbArena`. Every mutator sets `*dirty` so the deload path knows
        /// whether the on-disk image still matches; mutation routes are
        /// CLOSED (no `data()`, no iterators) so the flag cannot be bypassed.
        ///
        /// @param arena The LB's `LbArena`.
        /// @param dirty The aggregate's shared content-change state.
        explicit ArenaVector(LbArena* arena, DirtyState* dirty)
            : arena_(arena), dirty_(dirty) {
            assert(arena_ != nullptr);
            assert(dirty_ != nullptr);
        }

        ArenaVector(const ArenaVector&) = delete;
        ArenaVector(ArenaVector&&) = delete;
        ArenaVector& operator=(ArenaVector&&) = delete;

        /// @brief Drops the container's references; storage becomes holes the
        ///        compaction reclaims (the bump arena frees only its tail).
        ~ArenaVector() { release(); }

        /// @brief Number of elements.
        ///
        /// @details
        /// Asserts residency like the element accessors: a deloaded container
        /// reads as empty in RAM, and a silent size-0 answer turns loops over
        /// it into silent no-ops while the LB's resident side tables stay
        /// mutable (I-111).
        ///
        /// @return Logical element count.
        int32_t size() const {
            assert(arena_->resident()
                && "size() on a deloaded LB - read the recorded deload "
                   "counts for cold metadata");
            return size_;
        }

        /// @brief Whether the container is empty.
        ///
        /// @return `true` when `size() == 0`. Asserts residency like
        ///         `size()`.
        bool empty() const { return size() == 0; }

        /// @brief Approximate live byte footprint (elements + spine) — the
        ///        steward's fragmentation-gate input.
        ///
        /// @details
        /// Counts the live elements and the current spine (chunks +
        /// directory); deload holes (erased elements, superseded spine/dir
        /// runs, block padding) are the difference from the arena's used
        /// span. Pure arithmetic over the size fields; no residency needed.
        ///
        /// @return Live bytes the container's elements and spine occupy.
        int64_t liveBytes() const {
            return static_cast<int64_t>(size_)
                       * static_cast<int64_t>(sizeof(T))
                 + (static_cast<int64_t>(numChunks_) * kSpineChunk
                    + dirCap_) * static_cast<int64_t>(sizeof(ArenaOffset));
        }

        /// @brief Element access — read-only BY DESIGN (no mutable overload).
        ///
        /// @details
        /// Resolves the directory, the chunk, then the element: three
        /// `LbArena::resolve` calls (a shift + mask + lookup each). Asserts
        /// range and — via `resolve` — residency. There is deliberately NO
        /// mutable element access: in-place writes are impossible through the
        /// public surface, so the skip-unchanged deload contract is airtight.
        ///
        /// @param i Logical index in `[0, size())`.
        /// @return Const reference, valid while the LB stays resident, no
        ///         preceding element is erased, and no compaction runs.
        const T& operator[](int32_t i) const {
            assert(i >= 0 && i < size_);
            return *reinterpret_cast<const T*>(arena_->resolve(elemOffAt(i)));
        }

        /// @brief Append one element (a fresh scattered allocation), growing
        ///        the spine by a chunk (and the directory) on a boundary.
        ///
        /// @param value Element to copy in.
        void push_back(const T& value) {
            if (*dirty_ == DirtyState::Clean)
                *dirty_ = DirtyState::AppendedOnly;
            if (size_ == numChunks_ * kSpineChunk) addChunk();
            const ArenaOffset eOff = arena_->alloc(
                static_cast<int32_t>(sizeof(T)),
                static_cast<int32_t>(alignof(T)));
            std::memcpy(arena_->resolve(eOff), &value, sizeof(T));
            setElemOff(size_, eOff);
            ++size_;
        }

        /// @brief Erase the element at index `i`, shifting successors left
        ///        (the `std::vector::erase` contract).
        ///
        /// @details
        /// Shifts the SPINE offsets (4 bytes each), not the element payloads,
        /// so it is far cheaper than the paged predecessor; the erased
        /// element's payload becomes a hole the compaction reclaims. O(size -
        /// i) — fine for the cold maintenance paths (`wipeSubtree`-style
        /// back-to-front loops).
        ///
        /// @param i Logical index in `[0, size())`.
        void erase(int32_t i) {
            assert(i >= 0 && i < size_);
            *dirty_ = DirtyState::Restructured;
            for (int32_t j = i; j + 1 < size_; ++j)
                setElemOff(j, elemOffAt(j + 1));
            --size_;
        }

        /// @brief Drop all elements; keep the spine (directory + chunks) for
        ///        reuse.
        ///
        /// @details
        /// Element payloads (and any old content) become holes the compaction
        /// reclaims — the bump arena cannot free interior storage. The spine
        /// is retained so a refill does not re-grow it.
        void clear() {
            *dirty_ = DirtyState::Restructured;
            size_ = 0;
        }

        /// @brief `clear` plus dropping the spine reference — successor of the
        ///        swap-with-empty release idiom.
        void release() {
            *dirty_ = DirtyState::Restructured;
            size_ = 0;
            numChunks_ = 0;
            dirCap_ = 0;
            dirOff_ = kNullOffset;
        }

        /// @brief Deep copy (element-wise), including across two different LB
        ///        arenas — the LB-clone path.
        ///
        /// @param other Source vector (unchanged).
        /// @return `*this`.
        ArenaVector& operator=(const ArenaVector& other) {
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
        ArenaVector& operator=(const std::vector<T>& src) {
            clear();
            for (const T& value : src) push_back(value);
            return *this;
        }

        /// @brief Append the bytes of rows `[fromRow, size())` to a byte
        ///        buffer — the canonical element-stream dump.
        ///
        /// @details
        /// Emits each element's `sizeof(T)` bytes in logical order. Scattered
        /// storage forces an element-by-element walk, but the OUTPUT is
        /// byte-identical to the paged predecessor's page-span dump. Read-only.
        ///
        /// @param out     Destination byte buffer (appended to).
        /// @param fromRow First row to emit, in `[0, size()]`.
        void appendSpanBytes(std::vector<char>& out, int32_t fromRow) const {
            assert(fromRow >= 0 && fromRow <= size_);
            if (fromRow == size_) return;
            const std::size_t base = out.size();
            out.resize(base + static_cast<std::size_t>(size_ - fromRow)
                              * sizeof(T));
            char* dst = out.data() + base;
            for (int32_t i = fromRow; i < size_; ++i) {
                std::memcpy(dst, arena_->resolve(elemOffAt(i)), sizeof(T));
                dst += sizeof(T);
            }
        }

        /// @brief Append `rowCount` elements from a packed byte stream — the
        ///        bulk reload path.
        ///
        /// @details
        /// Reconstructs each element and `push_back`s it, rebuilding the
        /// scattered elements and the spine fresh — the deload image carries
        /// only logical content, never offsets, so reload re-derives them.
        ///
        /// @param bytes    Source stream of `rowCount * sizeof(T)` bytes.
        /// @param rowCount Elements to append; >= 0.
        void bulkAppendBytes(const char* bytes, int64_t rowCount) {
            assert(rowCount >= 0);
            *dirty_ = DirtyState::Restructured;
            for (int64_t r = 0; r < rowCount; ++r) {
                T value;
                std::memcpy(&value,
                            bytes + static_cast<std::size_t>(r) * sizeof(T),
                            sizeof(T));
                push_back(value);
            }
        }

    private:
        /// @brief The element offset stored at logical index `i` (resolve the
        ///        directory, then the chunk).
        ///
        /// @param i Logical index in `[0, size())`.
        /// @return The element's virtual offset.
        ArenaOffset elemOffAt(int32_t i) const {
            const ArenaOffset chunkOff =
                reinterpret_cast<const ArenaOffset*>(
                    arena_->resolve(dirOff_))[i >> kChunkShift];
            return reinterpret_cast<const ArenaOffset*>(
                arena_->resolve(chunkOff))[i & kChunkMask];
        }

        /// @brief Store element offset `off` at logical index `i`.
        ///
        /// @param i   Logical index whose chunk slot is already allocated.
        /// @param off The element's virtual offset.
        void setElemOff(int32_t i, ArenaOffset off) {
            const ArenaOffset chunkOff =
                reinterpret_cast<const ArenaOffset*>(
                    arena_->resolve(dirOff_))[i >> kChunkShift];
            reinterpret_cast<ArenaOffset*>(
                arena_->resolve(chunkOff))[i & kChunkMask] = off;
        }

        /// @brief Append one chunk to the spine, growing the directory first
        ///        when it is full.
        void addChunk() {
            if (numChunks_ == dirCap_) growDir();
            const ArenaOffset chunkOff = arena_->alloc(
                kSpineChunk * static_cast<int32_t>(sizeof(ArenaOffset)),
                static_cast<int32_t>(alignof(ArenaOffset)));
            reinterpret_cast<ArenaOffset*>(
                arena_->resolve(dirOff_))[numChunks_] = chunkOff;
            ++numChunks_;
        }

        /// @brief Grow the chunk directory by reallocation; the old run
        ///        becomes a hole the compaction reclaims.
        void growDir() {
            const int32_t newCap = (dirCap_ == 0) ? kInitialDirCap
                                                  : dirCap_ * 2;
            const ArenaOffset newDir = arena_->alloc(
                newCap * static_cast<int32_t>(sizeof(ArenaOffset)),
                static_cast<int32_t>(alignof(ArenaOffset)));
            if (dirCap_ > 0) {
                std::memcpy(arena_->resolve(newDir),
                            arena_->resolve(dirOff_),
                            static_cast<std::size_t>(numChunks_)
                                * sizeof(ArenaOffset));
            }
            dirOff_ = newDir;
            dirCap_ = newCap;
        }

        static constexpr int32_t kSpineChunk = 1024;  // offsets per chunk
        static constexpr int32_t kChunkShift = 10;    // log2(kSpineChunk)
        static constexpr int32_t kChunkMask = 1023;   // kSpineChunk - 1
        static constexpr int32_t kInitialDirCap = 4;

        LbArena* arena_;
        DirtyState* dirty_;          // aggregate's content-change state
        ArenaOffset dirOff_ = kNullOffset;  // directory of chunk offsets
        int32_t dirCap_ = 0;         // directory slots allocated
        int32_t numChunks_ = 0;      // chunks in use
        int32_t size_ = 0;           // logical element count
    };

}
