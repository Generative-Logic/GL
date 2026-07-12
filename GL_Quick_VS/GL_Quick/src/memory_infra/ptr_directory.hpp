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

namespace gl {

    /// @brief Paranoid-assert switch for the static-arena bookkeeping.
    ///
    /// @details
    /// `LbArena` is the substrate every per-LB container resolves through, so a
    /// silent bug here corrupts every container and surfaces far away. The
    /// always-on contract asserts (every method below) plus the coarse-seam
    /// `assertInvariants()` audits (NDEBUG-gated) catch most of it; this switch
    /// arms the EXTRA cost — the per-access asserts on the hottest paths
    /// (`LbArena::pageAt` / `resolve`) and the fine-seam `assertInvariants()`
    /// audits after every `allocPage` / `freePage` — that release builds gate off
    /// for RT. Unit tests and debug runs define it to 1; release leaves it 0.
    #ifndef GL_ARENA_PARANOID
    #define GL_ARENA_PARANOID 0
    #endif

    /// @brief Spilled-region fan-out of `PtrDirectory` — the number of pool data
    ///        blocks the wired root can address beyond the inline buffer, and so
    ///        the "assert, not recurse" ceiling.
    ///
    /// @details
    /// One MAIN-pool data block holds `262144 / 8 = 32768` `char*` entries. With
    /// `kArenaDirRootCap == 16` the spilled region addresses up to `16 * 32768 =
    /// 524288` entries on top of the inline buffer — i.e. far beyond any single LB
    /// (the whole 4 GiB main pool contains only 16384 data blocks). Exhaustion
    /// (a 17th spilled data block is needed) is a hard assert naming this constant
    /// (Rule 19 / I-19) — never a third directory level, never a fallback.
    constexpr int32_t kArenaDirRootCap = 16;

    /// @brief Inline-buffer size for the per-LB page table (`LbArena::pageTable_`).
    ///
    /// @details
    /// A page table with at most this many vids lives entirely in the
    /// `PtrDirectory` object (no pool block at all), so the numerous SMALL arenas
    /// — the per-worker scratch arenas, the transient routing-mail arenas, and the
    /// never-deloaded `persistentArena` backing a typical small `intToBeProved` —
    /// cost ZERO directory blocks and never pressure their pools. Only a large
    /// page table (the main deloadable arena, thousands of pages) spills to pool
    /// blocks, where the one directory block is negligible against its data. At
    /// `128` the inline cost is `128 * sizeof(char*) = 1 KiB` per page table; the
    /// inline buffer covers up to a 1 MiB page footprint (128 * 8 KiB pages).
    constexpr int32_t kArenaPageTableInline = 128;

    /// @brief Inline-buffer size for the per-LB block tables (`LbArena::blocks_`,
    ///        `pageBlocks_`) — small because block counts are O(arena bytes /
    ///        256 KiB), tiny for every realistic arena.
    constexpr int32_t kArenaBlockTableInline = 16;

    /// @brief A growable, mostly-inline `char*` array — the static replacement for
    ///        the heap `std::vector<char*>` bookkeeping inside `LbArena` (block
    ///        table, page table, carved-block list).
    ///
    /// @details
    /// `LbArena` is the bottom of the static-memory hierarchy: it manages the pool
    /// blocks every paged container's storage lives in. Its own bookkeeping —
    /// which physical block backs byte-bump index `i`, which physical page backs
    /// vid `v` — was the last per-LB heap carve-out. It cannot ride the page tier
    /// the way `PagedVector` / `PagedHashIndex` do (those resolve THROUGH the very
    /// page table this container IS — a bootstrap regress). So `PtrDirectory`
    /// keeps a small inline buffer for the common case and, only when that
    /// overflows, draws data blocks DIRECTLY from the `GlobalMemoryManager`
    /// (`acquireBlock` / `releaseBlock`) — the same grant path `LbArena` uses.
    ///
    /// **Small-buffer optimization.** The first `kInline` entries live in the
    /// inline array `inline_[kInline]` (part of the `LbArena` / `Memory` object,
    /// already pool-backed via `LbStore`), so a table that never exceeds `kInline`
    /// costs no separate pool block at all. Entries `[kInline, ...)` "spill" and
    /// pack contiguously across data blocks held by a small wired inline root
    /// `root_[kArenaDirRootCap]`; spilled entry `s = i - kInline` resolves as
    /// `root_[s >> shift_] + (s & mask_) * sizeof(char*)`. One level of
    /// indirection, NO recursion: the root IS the spilled directory, the data
    /// blocks are its leaves, and overflowing the root asserts (see
    /// `kArenaDirRootCap`).
    ///
    /// **Retain on shrink.** `pop_back` / `truncate` keep capacity (like a
    /// `std::vector`); spilled data blocks return to the pool only at `clear`
    /// (deload / teardown) or `shrinkToFit` (the coarse compaction seam). A
    /// per-operation shrink-then-grow (a scratch arena's `rewind` then refill)
    /// therefore never churns pool grants — both for speed and because the
    /// monotone grant ledger feeds the steward's deterministic deload decisions
    /// (I-106).
    ///
    /// **Determinism (I-107).** `PtrDirectory` stores opaque `char*` it never
    /// interprets or orders; the index `i` (a byte-bump block number or a vid) is
    /// the caller's deterministic, allocation-order handle, and the physical block
    /// identity the directory holds is invisible to every observable — the
    /// directory is never serialized into a deload image (it is rebuilt on reload,
    /// exactly like the `std::vector` it replaces).
    ///
    /// @tparam kInline Inline-buffer length; a table of at most `kInline` entries
    ///         holds no pool block.
    /// @invariant `numBlocks_ >= ceil(max(0, size_ - kInline) / entriesPerBlock_)`
    ///            (spilled data blocks are retained on shrink and returned only by
    ///            `clear` / `shrinkToFit`); root slots `[0, numBlocks_)` are
    ///            non-null pool blocks and `[numBlocks_, kArenaDirRootCap)` null.
    /// @invariant Entries may be null (a `pageTable_` slot for a freed vid is
    ///            null); `PtrDirectory` imposes no non-null entry contract — that
    ///            is the owning table's semantics, checked in
    ///            `LbArena::assertInvariants`.
    /// @see `LbArena` (the owner), `PagedHashIndex` (the page-tier-backed
    ///      mutable-slot sibling), `GlobalMemoryManager`, `kArenaDirRootCap`.
    template <int32_t kInline>
    class PtrDirectory {
        static_assert(kInline > 0, "PtrDirectory inline buffer must be positive");

    public:
        /// @brief Construct an UNBOUND directory (no pool, no geometry) — bound
        ///        later by `bind`.
        PtrDirectory() = default;

        /// @brief Bind to the global manager that grants this directory's spilled
        ///        blocks (lazy: holds nothing and computes no geometry until the
        ///        first spill).
        ///
        /// @param global The block source. Must be non-null.
        explicit PtrDirectory(GlobalMemoryManager* global) : global_(global) {
            assert(global_ != nullptr);
        }

        /// @brief Returns every held spilled data block to the global manager.
        ~PtrDirectory() { clear(); }

        PtrDirectory(const PtrDirectory&) = delete;
        PtrDirectory& operator=(const PtrDirectory&) = delete;
        PtrDirectory(PtrDirectory&&) = delete;
        PtrDirectory& operator=(PtrDirectory&&) = delete;

        /// @brief Lazily bind a default-constructed directory to a pool.
        ///
        /// @param global Pool to draw spilled blocks from.
        void bind(GlobalMemoryManager* global) {
            assert(global_ == nullptr && "PtrDirectory::bind on a bound directory");
            assert(global != nullptr && global->initialized());
            assert(size_ == 0 && numBlocks_ == 0);
            global_ = global;
        }

        /// @brief Number of entries currently stored.
        int32_t size() const { return size_; }

        /// @brief Whether the directory holds no entries.
        bool empty() const { return size_ == 0; }

        /// @brief Append one entry, spilling onto a fresh pool data block only
        ///        when the inline buffer is full and the current spilled block
        ///        fills.
        ///
        /// @details
        /// Asserts BEFORE a spill grant that the wired root has room
        /// (`numBlocks_ < kArenaDirRootCap`) — the "assert, not recurse" ceiling.
        ///
        /// @param entry The pointer to store (may be null — the owner's semantics
        ///              decide, e.g. a freed-vid slot).
        void push_back(char* entry) {
            if (size_ >= kInline) {
                ensureGeometry();
                const int32_t s = size_ - kInline;
                const int32_t blockIdx = s >> shift_;
                if (blockIdx >= numBlocks_) {
                    assert(blockIdx == numBlocks_
                        && "PtrDirectory spilled by more than one block in a push");
                    assert(numBlocks_ < kArenaDirRootCap
                        && "PtrDirectory exceeded kArenaDirRootCap spilled blocks "
                           "— raise kArenaDirRootCap (and re-measure the pool "
                           "budget); never a second directory level");
                    root_[numBlocks_] = global_->acquireBlock();
                    assert(root_[numBlocks_] != nullptr);
                    ++numBlocks_;
                }
            }
            writeAt(size_, entry);
            ++size_;
        }

        /// @brief Read entry `i`.
        ///
        /// @param i Entry index in `[0, size())`.
        /// @return The stored pointer (may be null).
        char* operator[](int32_t i) const {
            assert(i >= 0 && i < size_ && "PtrDirectory::operator[] out of range");
            return readAt(i);
        }

        /// @brief Read entry `i` WITHOUT the bounds assert — the hot-path accessor
        ///        for `LbArena::pageAt`.
        ///
        /// @details
        /// `LbArena::pageAt` is the hottest resolve in the prover and gates its own
        /// checks under `GL_ARENA_PARANOID` for RT; the asserting `operator[]`
        /// (whose assert is live even in release, since the build keeps asserts on)
        /// would tax that path on every page resolve. The caller guarantees `i` is
        /// a live index.
        ///
        /// @param i A live entry index in `[0, size())` (caller-guaranteed).
        char* peek(int32_t i) const { return readAt(i); }

        /// @brief Overwrite entry `i` in place — the operation `pageTable_`
        ///        compaction needs (rebind a vid to a new physical page).
        ///
        /// @param i Entry index in `[0, size())`.
        /// @param v The pointer to store (may be null).
        void set(int32_t i, char* v) {
            assert(i >= 0 && i < size_ && "PtrDirectory::set out of range");
            writeAt(i, v);
        }

        /// @brief The last entry.
        char* back() const {
            assert(size_ > 0 && "PtrDirectory::back on an empty directory");
            return readAt(size_ - 1);
        }

        /// @brief Drop the last entry, RETAINING capacity (like a `std::vector`).
        ///
        /// @details
        /// Decrements the count only — no pool churn on a shrink-then-grow (a
        /// scratch arena's `rewind` then refill via `LbArena::freePage`'s tail
        /// bump-back). Spilled blocks return to the pool only at `clear` /
        /// `shrinkToFit`.
        void pop_back() {
            assert(size_ > 0 && "PtrDirectory::pop_back on an empty directory");
            --size_;
        }

        /// @brief Truncate to `n` entries, RETAINING capacity.
        ///
        /// @param n New size in `[0, size()]`.
        void truncate(int32_t n) {
            assert(n >= 0 && n <= size_ && "PtrDirectory::truncate past the size");
            size_ = n;
        }

        /// @brief Release spilled data blocks the current size no longer needs —
        ///        the COARSE-seam reclaim (`LbArena::compactPages`, the steward's
        ///        reclaim), never the per-op shrink path (so scratch rewind stays
        ///        churn-free). After a discharge empties the page tier this returns
        ///        the last spilled directory block, so the LB ends at zero held
        ///        blocks.
        void shrinkToFit() {
            const int32_t spilled = (size_ > kInline) ? (size_ - kInline) : 0;
            const int32_t need =
                (spilled == 0) ? 0 : (((spilled - 1) >> shift_) + 1);
            while (numBlocks_ > need) {
                --numBlocks_;
                assert(root_[numBlocks_] != nullptr);
                global_->releaseBlock(root_[numBlocks_]);
                root_[numBlocks_] = nullptr;
            }
        }

        /// @brief Return every held spilled data block to the pool and reset to
        ///        empty (cached geometry kept; the inline buffer is abandoned in
        ///        place, costing no pool block).
        void clear() {
            for (int32_t b = 0; b < numBlocks_; ++b) {
                assert(root_[b] != nullptr);
                global_->releaseBlock(root_[b]);
                root_[b] = nullptr;
            }
            numBlocks_ = 0;
            size_ = 0;
        }

        /// @brief Spilled data blocks currently held (the owner adds these into
        ///        `LbArena::blocksHeld`; the inline buffer holds no pool block).
        int32_t blocksHeld() const { return numBlocks_; }

        /// @brief Deep structural self-audit — asserts the whole directory is
        ///        self-consistent. NDEBUG-gated (empty in release builds that
        ///        define NDEBUG; this project keeps asserts on).
        void assertInvariants() const {
#ifndef NDEBUG
            assert(size_ >= 0);
            assert(numBlocks_ >= 0 && numBlocks_ <= kArenaDirRootCap);
            const int32_t spilled = (size_ > kInline) ? (size_ - kInline) : 0;
            if (shift_ < 0) {                    // never spilled
                assert(numBlocks_ == 0);
            } else {
                assert(entriesPerBlock_ == (1 << shift_));
                assert(mask_ == entriesPerBlock_ - 1);
                const int32_t need =
                    (spilled == 0) ? 0 : (((spilled - 1) >> shift_) + 1);
                assert(numBlocks_ >= need
                    && "PtrDirectory has fewer spilled blocks than size needs");
                assert(spilled <= numBlocks_ * entriesPerBlock_);
            }
            for (int32_t b = 0; b < numBlocks_; ++b)
                assert(root_[b] != nullptr
                    && "PtrDirectory in-use root slot is null");
            for (int32_t b = numBlocks_; b < kArenaDirRootCap; ++b)
                assert(root_[b] == nullptr
                    && "PtrDirectory unused root slot is non-null");
#endif
        }

    private:
        /// @brief Read entry `i` (inline or spilled), no bounds check.
        char* readAt(int32_t i) const {
            if (i < kInline) return inline_[i];
            const int32_t s = i - kInline;
            return *reinterpret_cast<char* const*>(
                root_[s >> shift_]
                + static_cast<std::size_t>(s & mask_) * sizeof(char*));
        }

        /// @brief Write entry `i` (inline or spilled), no bounds check.
        void writeAt(int32_t i, char* v) {
            if (i < kInline) { inline_[i] = v; return; }
            const int32_t s = i - kInline;
            *reinterpret_cast<char**>(
                root_[s >> shift_]
                + static_cast<std::size_t>(s & mask_) * sizeof(char*)) = v;
        }

        /// @brief Compute the power-of-two entries-per-block split on first spill.
        ///
        /// @details
        /// Asserts `blockBytes` is a whole power-of-two multiple of `sizeof(char*)`
        /// so the spilled-slot split is one shift plus one mask — true by the
        /// pool's `isValidStaticMemoryConfig` (block a power of two) and
        /// `sizeof(char*) == 8`.
        void ensureGeometry() {
            if (shift_ >= 0) return;
            assert(global_ != nullptr && global_->initialized()
                && "PtrDirectory spill before the pool is initialized");
            const int32_t bb = global_->blockBytes();
            const int32_t per = bb / static_cast<int32_t>(sizeof(char*));
            assert(per > 0 && (per & (per - 1)) == 0
                && "static_block_bytes / sizeof(char*) must be a power of two");
            entriesPerBlock_ = per;
            mask_ = per - 1;
            int32_t s = 0;
            while ((1 << s) < per) ++s;
            shift_ = s;
        }

        GlobalMemoryManager* global_ = nullptr;   // null until bound
        char* inline_[kInline] = {};              // first kInline entries (no pool block)
        char* root_[kArenaDirRootCap] = {};       // spilled data blocks; null past numBlocks_
        int32_t size_ = 0;                         // entries stored (inline + spilled)
        int32_t numBlocks_ = 0;                    // spilled data blocks held
        int32_t entriesPerBlock_ = 0;              // blockBytes / sizeof(char*)
        int32_t shift_ = -1;                       // log2(entriesPerBlock_); -1 = no spill yet
        int32_t mask_ = 0;                         // entriesPerBlock_ - 1
    };

}
