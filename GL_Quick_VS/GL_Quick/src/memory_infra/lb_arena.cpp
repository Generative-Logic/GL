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

#include "lb_arena.hpp"

#include "dirty_state.hpp"
#include "paged_hash_index.hpp"
#include "paged_vector.hpp"
#include "../parameters.hpp"

#include <cstring>

namespace gl {

    /// @brief Bind to the global manager that grants this arena's blocks.
    ///
    /// @details
    /// Lazy: holds zero blocks and computes no geometry until the first
    /// `alloc` — transient LB objects must not consume pool blocks (or touch
    /// the not-yet-initialized pool) merely by existing.
    ///
    /// @param global The block source (process-wide in production; a private
    ///               instance in unit tests). Must be non-null.
    LbArena::LbArena(GlobalMemoryManager* global)
        : global_(global), blocks_(global), pageBlocks_(global),
          pageTable_(global) {
        assert(global_ != nullptr);
    }

    /// @brief Returns every held block to the global manager.
    LbArena::~LbArena() {
        releaseAll();
    }

    /// @brief Compute the block geometry from the global manager on first
    ///        use.
    ///
    /// @details
    /// Deferred because LB objects can be constructed before the pool is
    /// initialized (the prover's `Memory` members are built before the config
    /// parse). Asserts the block size is a power of two — the shift / mask
    /// split requires it.
    void LbArena::ensureGeometry() {
        if (blockBytes_ != 0) return;
        assert(global_->initialized()
            && "LbArena::alloc before the pool is initialized");
        const int32_t bb = global_->blockBytes();
        assert(bb > 0 && (bb & (bb - 1)) == 0
            && "static_block_bytes must be a power of two for the arena "
               "shift/mask split");
        blockBytes_ = bb;
        blockMask_ = static_cast<uint32_t>(bb - 1);
        int32_t s = 0;
        while ((static_cast<int32_t>(1) << s) < bb) ++s;
        blockShift_ = s;
    }

    /// @brief Acquire blocks until `blockIndex` is backed by storage.
    ///
    /// @details
    /// Draws from the pool via `acquireBlock` (LB-side accounting); pool
    /// exhaustion asserts naming `static_pool_bytes`, never spills to the heap.
    ///
    /// @param blockIndex The block index the cursor has reached.
    void LbArena::ensureBlock(int64_t blockIndex) {
        while (static_cast<int64_t>(blocks_.size()) <= blockIndex) {
            blocks_.push_back(global_->acquireBlock());
        }
    }

    /// @brief Bump-allocate `bytes`, aligned to `align`, and return the new
    ///        allocation's virtual offset.
    ///
    /// @details
    /// Aligns the cursor up to `align`, pads to the next block boundary if
    /// the request would straddle the current block, acquires fresh blocks
    /// from the global manager as the cursor crosses block boundaries, then
    /// advances the cursor by `bytes`. Asserts residency, a positive length
    /// that fits one block, and a power-of-two `align` no greater than the
    /// block alignment.
    ///
    /// @param bytes Exact length of the allocation; `0 < bytes <= blockBytes`.
    /// @param align Power-of-two alignment of the returned offset, in
    ///              `[1, 16]`. Defaults to 1.
    /// @return The virtual offset of the allocation.
    ArenaOffset LbArena::alloc(int32_t bytes, int32_t align) {
        assert(resident_ && "LbArena::alloc on a deloaded arena");
        assert(bytes > 0);
        assert(align >= 1 && (align & (align - 1)) == 0 && align <= 16
            && "LbArena::alloc alignment must be a power of two in [1,16]");
        ensureGeometry();
        assert(bytes <= blockBytes_
            && "single arena allocation exceeds a block — raise "
               "static_block_bytes or rethink the caller");
        // Align the cursor up to the requested alignment.
        const uint32_t alignMask = static_cast<uint32_t>(align) - 1u;
        cursor_ = (cursor_ + alignMask) & ~alignMask;
        // No-straddle: pad to the next block boundary if the request would
        // not fit the current block's remainder.
        const uint32_t within = cursor_ & blockMask_;
        if (static_cast<int64_t>(within) + bytes > blockBytes_)
            cursor_ += (static_cast<uint32_t>(blockBytes_) - within);
        const int64_t block = static_cast<int64_t>(cursor_) >> blockShift_;
        ensureBlock(block);
        const ArenaOffset off = cursor_;
        assert(off != kNullOffset
            && "LbArena offset space exhausted (4 GiB per LB)");
        cursor_ += static_cast<uint32_t>(bytes);
        if (static_cast<int64_t>(cursor_) > peakUsedBytes_)
            peakUsedBytes_ = static_cast<int64_t>(cursor_);
        return off;
    }

    /// @brief Tail-free back to a previously captured cursor position.
    ///
    /// @details
    /// Rolls the cursor back to `off` (last-in-first-out) and poisons the
    /// reclaimed span with `kArenaPoisonByte`, block by block. Retained
    /// blocks stay retained — the next `alloc` reuses them. Legal only
    /// backwards; a no-op when `off == cursor()`.
    ///
    /// @param off A cursor value captured by an earlier `cursor()`.
    void LbArena::popTo(ArenaOffset off) {
        assert(resident_ && "LbArena::popTo on a deloaded arena");
        assert(off <= cursor_ && "LbArena::popTo forwards past the cursor");
        if (off == cursor_) return;
        ArenaOffset p = off;
        while (p < cursor_) {
            const int64_t block = static_cast<int64_t>(p) >> blockShift_;
            const uint32_t within = p & blockMask_;
            const int64_t blockEnd64 = (block + 1)
                << static_cast<int64_t>(blockShift_);
            const ArenaOffset spanEnd =
                (static_cast<int64_t>(cursor_) < blockEnd64)
                    ? cursor_
                    : static_cast<ArenaOffset>(blockEnd64);
            std::memset(blocks_.peek(static_cast<int32_t>(block)) + within,
                        kArenaPoisonByte,
                        static_cast<std::size_t>(spanEnd - p));
            p = spanEnd;
        }
        cursor_ = off;
    }

    /// @brief Carve or reuse one physical page for `allocPage`.
    ///
    /// @details
    /// Pops the free-list when non-empty; otherwise acquires a fresh block
    /// (`acquireBlock`) and carves it into `pageBytes` pages, pushing them onto
    /// the free-list, then pops one.
    ///
    /// @return A physical page free for binding to a fresh vid.
    char* LbArena::takePage() {
        const int32_t pb = global_->pageBytes();
        if (freeHead_ == nullptr) {
            ensureGeometry();
            char* block = global_->acquireBlock();
            pageBlocks_.push_back(block);
            const int32_t ppb = global_->blockBytes() / pb;
            // Thread the fresh pages onto the intrusive LIFO: the next-free link
            // lives in the LAST sizeof(char*) bytes of each page, so freePage's
            // poison still covers offset 0 (the use-after-free tripwire the
            // scratch strings rely on).
            for (int32_t p = 0; p < ppb; ++p) {
                char* page = block + static_cast<std::ptrdiff_t>(p) * pb;
                *reinterpret_cast<char**>(page + pb - sizeof(char*)) = freeHead_;
                freeHead_ = page;
                ++freePageCount_;
            }
        }
        char* page = freeHead_;
        freeHead_ = *reinterpret_cast<char**>(page + pb - sizeof(char*));
        --freePageCount_;
        return page;
    }

    /// @brief Allocate one page and return its virtual id (vid).
    ///
    /// @details
    /// Binds the next vid to a physical page from `takePage` (free-list reuse
    /// or a fresh carve). Asserts residency.
    ///
    /// @return The new page's vid.
    int32_t LbArena::allocPage() {
        assert(resident_ && "LbArena::allocPage on a deloaded arena");
        ensureGeometry();
        pageTable_.push_back(takePage());
        ++livePages_;
        const int32_t vid = pageTable_.size() - 1;
#if GL_ARENA_PARANOID
        assertInvariants();
#endif
        return vid;
    }

    /// @brief Free the page at `vid`, returning it to the free-list.
    ///
    /// @details
    /// Poisons the page so a stale read announces itself, parks it on the
    /// free-list, and marks the vid dead. Freeing the tail vid (and any freed
    /// vids exposed beneath it) rolls the page cursor back — the page-level
    /// bump-back the scratch per-scope windows rely on; an interior free leaves a
    /// hole the free-list fills later. Asserts residency and vid liveness.
    ///
    /// @param vid A vid returned by `allocPage` and not yet freed.
    void LbArena::freePage(int32_t vid) {
        assert(resident_ && "LbArena::freePage on a deloaded arena");
        assert(vid >= 0 && vid < pageTable_.size()
            && "LbArena::freePage on an out-of-range vid");
        char* page = pageTable_[vid];
        assert(page != nullptr
            && "LbArena::freePage on an already-freed vid");
        const int32_t pb = global_->pageBytes();
        std::memset(page, kArenaPoisonByte, static_cast<std::size_t>(pb));
        // Park on the intrusive LIFO free-list; the link lives in the page's LAST
        // sizeof(char*) bytes so the poison still covers offset 0.
        *reinterpret_cast<char**>(page + pb - sizeof(char*)) = freeHead_;
        freeHead_ = page;
        ++freePageCount_;
        pageTable_.set(vid, nullptr);
        --livePages_;
        // Tail bump-back: drop trailing freed vids so the page cursor rolls
        // back (the scratch per-scope reclaim shrinks the id space LIFO).
        while (!pageTable_.empty() && pageTable_.back() == nullptr)
            pageTable_.pop_back();
#if GL_ARENA_PARANOID
        assertInvariants();
#endif
    }

    /// @brief In-place page compaction: pack the live pages onto the
    ///        contiguous prefix of the held blocks and return the emptied
    ///        blocks to the pool — the background steward's reclaim.
    ///
    /// @details
    /// Live pages, taken in ascending-vid order, are moved onto slots
    /// `0..live-1` of the block list (slot s = block `s / pagesPerBlock`,
    /// page `s % pagesPerBlock`, blocks in grant order) — one `memcpy` per
    /// moved page, applied with backward chain-walking (a chain of out-of-place
    /// targets moves with no scratch; a pure cycle buffers one page in a
    /// scratch buffer). Blocks past `ceil(live / pagesPerBlock)` then hold no
    /// live page and go back to the pool; the free-list is rebuilt as the tail
    /// slots of the last kept block.
    ///
    /// CONTENT-INVISIBLE: only a vid's physical binding moves; the vid values,
    /// the live-page count, and every page's bytes are unchanged. Containers
    /// address pages by vid and resolve through `pageAt` on every access, so
    /// none of them needs touching or a pointer-cache refresh
    /// (I-107). MUST run under exclusive LB
    /// access (steward claim or single-threaded barrier), never mid-burst.
    ///
    /// @return Number of blocks returned to the global pool.
    int64_t LbArena::compactPages(LbArena& scr) {
        assert(resident_ && "LbArena::compactPages on a deloaded arena");
        if (pageBlocks_.empty()) return 0;
        const int32_t pageBytesN = global_->pageBytes();
        const int32_t ppb = global_->blockBytes() / pageBytesN;
        const int32_t totalSlots =
            static_cast<int32_t>(pageBlocks_.size()) * ppb;

        // All scratch lives on the caller's scratch arena `scr` — drawn from an
        // independent never-deloaded pool, NOT the deloadable pool being
        // compacted, so the reclaim never depends on the pool it frees.
        DirtyState scrDirty = DirtyState::Clean;   // dummy: scratch never deloads

        // Live vids ascending; their target slots are their ranks.
        PagedVector<int32_t> liveVids(&scr, &scrDirty);
        for (int32_t vid = 0; vid < pageTable_.size(); ++vid)
            if (pageTable_[vid] != nullptr)
                liveVids.push_back(vid);
        const int32_t live = liveVids.size();
        assert(live == livePages_);

        const auto ptrOfSlot = [&](int32_t s) -> char* {
            return pageBlocks_[s / ppb]
                   + static_cast<std::ptrdiff_t>(s % ppb) * pageBytesN;
        };
        const auto slotOfPtr = [&](const char* p) -> int32_t {
            for (int32_t b = 0; b < pageBlocks_.size(); ++b) {
                const std::ptrdiff_t off = p - pageBlocks_[b];
                if (off >= 0
                    && off < static_cast<std::ptrdiff_t>(ppb) * pageBytesN)
                    return b * ppb
                           + static_cast<int32_t>(off / pageBytesN);
            }
            assert(false && "page pointer outside the arena's blocks");
            return -1;
        };

        // rankOfVid: vid -> its rank among live vids. Only live vids are read
        // (sigma reads it for a live vid), so the zero default is never observed
        // — no fill needed.
        PagedHashIndex rankOfVid(&scr);
        rankOfVid.reset(pageTable_.size());
        for (int32_t k = 0; k < live; ++k)
            rankOfVid.set(liveVids[k], k);
        // vidAtSlot: slot -> occupying vid. reset() zeroes, so 0 means "empty"
        // and an occupied slot stores vid+1 (the heap version's -1 sentinel,
        // remapped onto the zero default).
        PagedHashIndex vidAtSlot(&scr);
        vidAtSlot.reset(totalSlots);
        for (int32_t k = 0; k < live; ++k)
            vidAtSlot.set(slotOfPtr(pageTable_[liveVids[k]]), liveVids[k] + 1);

        const auto sigma = [&](int32_t s) -> int32_t {
            const int32_t vid = vidAtSlot.at(s) - 1;
            assert(vid >= 0);
            return rankOfVid.at(vid);
        };
        const auto applyMove = [&](int32_t from, int32_t to) {
            const int32_t vid = vidAtSlot.at(from) - 1;
            assert(vid >= 0 && vidAtSlot.at(to) == 0);
            std::memcpy(ptrOfSlot(to), ptrOfSlot(from),
                        static_cast<std::size_t>(pageBytesN));
            pageTable_.set(vid, ptrOfSlot(to));
            vidAtSlot.set(to, vid + 1);
            vidAtSlot.set(from, 0);
        };

        // Paths: occupied slots outside the target prefix sink into the prefix.
        for (int32_t s = live; s < totalSlots; ++s) {
            if (vidAtSlot.at(s) == 0) continue;
            PagedVector<int32_t> path(&scr, &scrDirty);
            path.push_back(s);
            for (;;) {
                const int32_t nxt = sigma(path[path.size() - 1]);
                assert(nxt < live);
                path.push_back(nxt);
                if (vidAtSlot.at(nxt) == 0) break;
            }
            for (int32_t i = path.size() - 1; i-- > 0;)
                applyMove(path[i], path[i + 1]);
        }

        // Cycles inside the prefix: buffer one page in a scratch page, rotate.
        const int32_t rotVid = scr.allocPage();
        char* rotBuf = scr.pageAt(rotVid);
        for (int32_t k = 0; k < live; ++k) {
            const int32_t occupant = vidAtSlot.at(k) - 1;
            assert(occupant >= 0
                && "free prefix slot after the path pass — a live page "
                   "vanished");
            if (occupant == liveVids[k]) continue;
            PagedVector<int32_t> cyc(&scr, &scrDirty);
            cyc.push_back(k);
            for (;;) {
                const int32_t nxt = sigma(cyc[cyc.size() - 1]);
                if (nxt == k) break;
                cyc.push_back(nxt);
            }
            const int32_t tailSlot = cyc[cyc.size() - 1];
            const int32_t tailVid = vidAtSlot.at(tailSlot) - 1;
            std::memcpy(rotBuf, ptrOfSlot(tailSlot),
                        static_cast<std::size_t>(pageBytesN));
            vidAtSlot.set(tailSlot, 0);
            for (int32_t i = cyc.size() - 1; i-- > 0;)
                applyMove(cyc[i], cyc[i + 1]);
            std::memcpy(ptrOfSlot(k), rotBuf,
                        static_cast<std::size_t>(pageBytesN));
            pageTable_.set(tailVid, ptrOfSlot(k));
            vidAtSlot.set(k, tailVid + 1);
        }
        scr.freePage(rotVid);

        // Blocks past the packed prefix hold no live page — return them.
        const int32_t keptBlocks = (live + ppb - 1) / ppb;
        const int64_t reclaimed =
            static_cast<int64_t>(pageBlocks_.size()) - keptBlocks;
        for (int32_t b = keptBlocks; b < pageBlocks_.size(); ++b) {
            global_->releaseBlock(pageBlocks_[b]);
        }
        pageBlocks_.truncate(keptBlocks);
        // Rebuild the intrusive free-list: the slots past the live prefix in the
        // kept blocks, ascending — deterministic from the live count alone.
        freeHead_ = nullptr;
        freePageCount_ = 0;
        for (int32_t s = live; s < keptBlocks * ppb; ++s) {
            char* page = ptrOfSlot(s);
            *reinterpret_cast<char**>(page + pageBytesN - sizeof(char*)) =
                freeHead_;
            freeHead_ = page;
            ++freePageCount_;
        }
        // Coarse-seam reclaim: a discharged / heavily-shrunk page tier returns
        // its now-unused directory blocks here too (the per-op pop_back /
        // truncate retain them to keep scratch rewind churn-free).
        pageTable_.shrinkToFit();
        pageBlocks_.shrinkToFit();
        assertInvariants();
        return reclaimed;
    }

    /// @brief Flag the arena as deloaded to SSD.
    ///
    /// @details
    /// Asserts `releaseAll` already ran (no blocks held) — the flag records a
    /// completed deload, it does not perform one.
    void LbArena::markDeloaded() {
        assert(resident_ && "markDeloaded on an already deloaded arena");
        assert(blocks_.empty() && pageBlocks_.empty()
            && pageTable_.empty() && freeHead_ == nullptr
            && "markDeloaded before releaseAll returned the blocks");
        resident_ = false;
    }

    /// @brief Flag the arena as resident again (reload completed).
    void LbArena::markResident() {
        assert(!resident_ && "markResident on an arena that is resident");
        resident_ = true;
    }

    /// @brief Return every held block to the global manager and reset the
    ///        cursor to empty (cached geometry is kept).
    ///
    /// @details
    /// Legal at deload or teardown, when every container's offsets and page
    /// ids are simultaneously rebuilt (reload) or discarded (destruction), and
    /// at every scratch worker-task end (the per-task release). Returns BOTH
    /// the byte-bump blocks and the page-tier blocks via `releaseBlock`; the
    /// generation bumps so any scratch view from before the release asserts (a
    /// no-op for the per-LB arena, which has no generation consumers).
    void LbArena::releaseAll() {
        for (int32_t b = 0; b < blocks_.size(); ++b)
            global_->releaseBlock(blocks_[b]);
        for (int32_t b = 0; b < pageBlocks_.size(); ++b)
            global_->releaseBlock(pageBlocks_[b]);
        blocks_.clear();         // each clear() also returns that table's own
        pageBlocks_.clear();     // spilled directory blocks
        pageTable_.clear();
        freeHead_ = nullptr;     // the free pages lived in pageBlocks_ (returned)
        freePageCount_ = 0;
        livePages_ = 0;
        cursor_ = 0;
        ++generation_;
    }

    /// @brief Lazily bind a default-constructed arena to a pool (see the
    ///        header for the contract).
    ///
    /// @param global Pool to draw blocks from.
    void LbArena::bind(GlobalMemoryManager* global) {
        assert(global_ == nullptr && "LbArena::bind on a bound arena");
        assert(global != nullptr && global->initialized());
        global_ = global;
        blocks_.bind(global);
        pageBlocks_.bind(global);
        pageTable_.bind(global);
        // Geometry and blocks are acquired lazily on first alloc.
    }

    /// @brief Allocate `bytes` contiguously and return a writable pointer; the
    ///        scratch fill-then-wrap allocation (see the header for the
    ///        contract).
    ///
    /// @details
    /// A byte-bump `alloc` resolved to a `char*`. The byte tier (`cursor_` /
    /// `blocks_`) is independent of the page tier (`allocPage`), so scratch
    /// fill never collides with paged containers on the same arena. `alloc`
    /// asserts residency and a positive length that fits one block, pads to the
    /// next block boundary rather than straddle (one contiguous run), and
    /// advances `cursor_` / `peakUsedBytes_` — the liveness measure the scratch
    /// string views read through `usedBytes`.
    ///
    /// @param bytes Exact length; `0 < bytes <= blockBytes`.
    /// @return Pointer to `bytes` writable bytes.
    char* LbArena::allocBytes(int32_t bytes) {
        return resolve(alloc(bytes));
    }

    /// @brief Stack-like reclamation back to a captured position; the freed
    ///        span is poisoned.
    ///
    /// @details
    /// Asserts the mark belongs to the current generation (rewinding across a
    /// `reset` is a lifecycle bug). Rolls the byte cursor back to the mark and
    /// poisons the reclaimed span (`popTo`, so the freed bytes read loud
    /// garbage — the scratch-string rewind tripwire); retained blocks stay
    /// retained for the next `allocBytes`. Byte-bump throughout: the page tier
    /// is never the scratch substrate.
    ///
    /// @param m A `mark()` taken in the current generation.
    void LbArena::rewind(const Mark& m) {
        assert(m.generation == generation_
            && "LbArena::rewind across a reset — the mark's scope is dead");
        popTo(m.cursor);
    }

    /// @brief Wholesale per-scope reclamation: roll the byte cursor to 0,
    ///        poison everything used, bump the generation so every outstanding
    ///        view asserts. Blocks stay retained.
    ///
    /// @details
    /// `popTo(0)` rolls the byte cursor to the front and poisons every used
    /// byte (loud garbage for stragglers) — byte-bump throughout — keeping the
    /// blocks for reuse. The generation bump then invalidates every outstanding
    /// scratch string view.
    void LbArena::reset() {
        popTo(0);
        ++generation_;
    }

    /// @brief Deep structural self-audit of the arena's bookkeeping — byte-bump
    ///        and page tier (see the header for the full contract).
    void LbArena::assertInvariants() const {
#ifndef NDEBUG
        pageTable_.assertInvariants();
        blocks_.assertInvariants();
        pageBlocks_.assertInvariants();
        // Byte-bump: the held blocks exactly span the cursor.
        if (blockBytes_ != 0) {
            const int32_t needB = (cursor_ == 0) ? 0
                : (static_cast<int32_t>((cursor_ - 1) >> blockShift_) + 1);
            assert(blocks_.size() == needB
                && "LbArena::assertInvariants: byte-bump blocks != cursor span");
        } else {
            assert(blocks_.size() == 0 && cursor_ == 0);
        }
        // livePages_ equals the non-null page-table entries; the tail is never a
        // freed vid (freePage pops trailing nulls).
        const int32_t hw = pageTable_.size();
        int32_t live = 0;
        for (int32_t v = 0; v < hw; ++v)
            if (pageTable_[v] != nullptr) ++live;
        assert(live == livePages_
            && "LbArena::assertInvariants: livePages_ != non-null page count");
        if (hw > 0)
            assert(pageTable_[hw - 1] != nullptr
                && "LbArena::assertInvariants: page-table tail is a freed vid");
        // The intrusive free-list length matches the counter (the next-free link
        // lives in each free page's last sizeof(char*) bytes).
        const int32_t pb = (global_ != nullptr) ? global_->pageBytes() : 0;
        const auto nextFree = [pb](char* page) -> char* {
            return *reinterpret_cast<char**>(page + pb - sizeof(char*));
        };
        int32_t fc = 0;
        for (char* p = freeHead_; p != nullptr; p = nextFree(p))
            ++fc;
        assert(fc == freePageCount_
            && "LbArena::assertInvariants: free-list length != freePageCount_");
        // Every carved page is accounted: live (in the table) or free (on the list).
        if (blockBytes_ != 0) {
            const int32_t ppb = blockBytes_ / pb;
            assert(static_cast<int64_t>(livePages_) + freePageCount_
                       == static_cast<int64_t>(pageBlocks_.size()) * ppb
                && "LbArena::assertInvariants: carved pages != live + free");
        }
#if GL_ARENA_PARANOID
        // Heavy cross-checks (O(pages^2 + pages*blocks), paranoid-only): every
        // live and free page is distinct, page-aligned, and inside a carved
        // block; no page is both free and live; no two vids alias one page.
        const int32_t blockBytesN = (global_ != nullptr) ? global_->blockBytes() : 0;
        const auto inCarvedBlock = [&](const char* page) -> bool {
            for (int32_t b = 0; b < pageBlocks_.size(); ++b) {
                char* block = pageBlocks_[b];
                const std::ptrdiff_t off = page - block;
                if (off >= 0 && off < static_cast<std::ptrdiff_t>(blockBytesN))
                    return (off % pb) == 0;
            }
            return false;
        };
        for (int32_t v = 0; v < hw; ++v) {
            char* page = pageTable_[v];
            if (page == nullptr) continue;
            assert(inCarvedBlock(page)
                && "LbArena::assertInvariants: live page outside a carved block");
            for (int32_t w = v + 1; w < hw; ++w)
                assert(pageTable_[w] != page
                    && "LbArena::assertInvariants: two vids alias one page");
            for (char* f = freeHead_; f != nullptr; f = nextFree(f))
                assert(f != page
                    && "LbArena::assertInvariants: page is both live and free");
        }
        for (char* f = freeHead_; f != nullptr; f = nextFree(f))
            assert(inCarvedBlock(f)
                && "LbArena::assertInvariants: free page outside a carved block");
#endif
#endif
    }

}
