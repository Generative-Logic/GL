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

#include <algorithm>
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
    /// Pops the free-vid LIFO when a freed vid is available
    /// (D-228) — the dead slot itself stores the chain link —
    /// and binds it to a physical page from `takePage` (free-list reuse or a
    /// fresh carve); otherwise binds the next tail vid. The vid space is
    /// therefore bounded by the live-page high water, and reuse is
    /// deterministic (same allocation/free sequence, same vids). Asserts
    /// residency.
    ///
    /// @return The page's vid.
    int32_t LbArena::allocPage() {
        assert(resident_ && "LbArena::allocPage on a deloaded arena");
        ensureGeometry();
        int32_t vid;
        if (freeVidHead_ != -1) {
            vid = freeVidHead_;
            freeVidHead_ = deadSlotNextVid(pageTable_[vid]);
            pageTable_.set(vid, takePage());
        } else {
            pageTable_.push_back(takePage());
            vid = pageTable_.size() - 1;
        }
        ++livePages_;
#if GL_ARENA_PARANOID
        assertInvariants();
#endif
        return vid;
    }

    /// @brief Free the page at `vid`, returning it to the free-list.
    ///
    /// @details
    /// Poisons the page so a stale read announces itself, parks the physical
    /// page on the free-list, and pushes the vid onto the free-vid LIFO — the
    /// dead slot stores the tagged chain link (D-228), so the
    /// next `allocPage` reuses the vid instead of ratcheting the table.
    /// Asserts residency and vid liveness.
    ///
    /// @param vid A vid returned by `allocPage` and not yet freed.
    void LbArena::freePage(int32_t vid) {
        assert(resident_ && "LbArena::freePage on a deloaded arena");
        assert(vid >= 0 && vid < pageTable_.size()
            && "LbArena::freePage on an out-of-range vid");
        char* page = pageTable_[vid];
        assert(isLivePageSlot(page)
            && "LbArena::freePage on an already-freed vid");
        const int32_t pb = global_->pageBytes();
        std::memset(page, kArenaPoisonByte, static_cast<std::size_t>(pb));
        // Park on the intrusive LIFO free-list; the link lives in the page's LAST
        // sizeof(char*) bytes so the poison still covers offset 0.
        *reinterpret_cast<char**>(page + pb - sizeof(char*)) = freeHead_;
        freeHead_ = page;
        ++freePageCount_;
        // Push the vid onto the free-vid LIFO: the dead slot IS the chain node.
        pageTable_.set(vid, deadSlotEncode(freeVidHead_));
        freeVidHead_ = vid;
        --livePages_;
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
            if (isLivePageSlot(pageTable_[vid]))
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

    /// @brief Bulk-acquire `count` blocks into a block directory — the raw
    ///        reload's one-mutex-per-chunk grant (see the header contract).
    ///
    /// @param dir   The block directory (`blocks_` or `pageBlocks_`).
    /// @param count Blocks to acquire and append; >= 0 (0 = no-op).
    void LbArena::acquireInto(PtrDirectory<kArenaBlockTableInline>& dir,
                              int32_t count) {
        assert(resident_ && "acquireInto on a deloaded arena");
        assert(count >= 0);
        // Fixed stack chunk: one mutex acquisition per chunk, and directory
        // spill (which re-enters the manager) runs AFTER the lock is released.
        // 32 blocks = one full 256 KiB block-of-pages worth; every real LB
        // takes a single chunk.
        constexpr int32_t kChunk = 32;
        char* chunk[kChunk];
        int32_t got = 0;
        while (got < count) {
            const int32_t take = std::min(kChunk, count - got);
            global_->acquireBlocks(take, chunk);
            for (int32_t i = 0; i < take; ++i)
                dir.push_back(chunk[i]);
            got += take;
        }
    }

    /// @brief Bulk-return every data block a directory holds — one mutex batch
    ///        per chunk instead of one per block (see the header contract).
    ///
    /// @param dir The block directory whose entries to return.
    void LbArena::releaseFrom(PtrDirectory<kArenaBlockTableInline>& dir) {
        const int32_t n = dir.size();
        if (n == 0) return;
        constexpr int32_t kChunk = 32;
        char* chunk[kChunk];
        int32_t done = 0;
        while (done < n) {
            const int32_t take = std::min(kChunk, n - done);
            for (int32_t i = 0; i < take; ++i)
                chunk[i] = dir.peek(done + i);
            global_->releaseBlocks(chunk, take);
            done += take;
        }
    }

    /// @brief The arena's raw-image shape — the header scalars (see the header
    ///        contract).
    ///
    /// @return The `RawShape` the raw dumper stamps into the v4 header.
    LbArena::RawShape LbArena::rawShape() const {
        assert(resident_ && "rawShape on a deloaded arena");
        return RawShape{ global_->blockBytes(), global_->pageBytes(),
                         cursor_, pageTable_.size(), livePages_ };
    }

    /// @brief Fill `bitmap` with one bit per vid — set iff the vid is live (see
    ///        the header contract; the whole-range wrapper over the chunked
    ///        `fillLiveBitmapRange`).
    ///
    /// @param bitmap      Caller buffer of `bitmapBytes` bytes.
    /// @param bitmapBytes Must equal `ceil(vidCount / 8)`.
    void LbArena::fillLiveBitmap(unsigned char* bitmap,
                                 int32_t bitmapBytes) const {
        const int32_t hw = pageTable_.size();
        assert(bitmapBytes == (hw + 7) / 8
            && "fillLiveBitmap buffer size != ceil(vidCount/8)");
        fillLiveBitmapRange(bitmap, 0, hw);
    }

    /// @brief Fill a bit-chunk of the live-vid bitmap for vids
    ///        `[startVid, startVid + vidSpan)` (see the header contract).
    ///
    /// @details
    /// Bit `i` of the chunk = vid `startVid + i` live; `startVid` must be a
    /// multiple of 8 so the chunk bytes concatenate to the exact whole-range
    /// bitmap the single-buffer fill produces.
    ///
    /// @param bitmap   Caller buffer of `ceil(vidSpan / 8)` bytes.
    /// @param startVid First vid of the chunk; a multiple of 8; >= 0.
    /// @param vidSpan  Vids in the chunk; >= 0.
    void LbArena::fillLiveBitmapRange(unsigned char* bitmap,
                                      int32_t startVid,
                                      int32_t vidSpan) const {
        assert(resident_ && "fillLiveBitmapRange on a deloaded arena");
        assert(startVid >= 0 && vidSpan >= 0);
        assert((startVid & 7) == 0
            && "fillLiveBitmapRange: chunk must start on a byte boundary");
        assert(startVid + vidSpan <= pageTable_.size()
            && "fillLiveBitmapRange: chunk exceeds the vid high-water");
        std::memset(bitmap, 0, static_cast<std::size_t>((vidSpan + 7) / 8));
        for (int32_t i = 0; i < vidSpan; ++i)
            if (isLivePageSlot(pageTable_.peek(startVid + i)))
                bitmap[i >> 3] |=
                    static_cast<unsigned char>(1u << (i & 7));
    }

    /// @brief Rebuild a FRESH page tier that binds ascending live vids to
    ///        consecutive dense pages — the single-shot wrapper over the
    ///        staged restore (see the header contract).
    ///
    /// @param vidCount       The vid high-water to reproduce (holes included).
    /// @param liveBitmap     `ceil(vidCount/8)` bytes; bit `v` set iff live.
    /// @param liveCount      Set bits in `liveBitmap`; the pages to carve.
    /// @param byteBumpCursor The byte-bump span to reproduce (expected 0).
    void LbArena::restoreForRawLoad(int32_t vidCount,
                                    const unsigned char* liveBitmap,
                                    int32_t liveCount,
                                    ArenaOffset byteBumpCursor) {
        restoreForRawLoadBegin(vidCount, liveCount, byteBumpCursor);
        int32_t slot = 0;
        if (vidCount > 0)
            slot = restoreForRawLoadChunk(liveBitmap, 0, vidCount, 0);
        restoreForRawLoadEnd(vidCount, slot);
    }

    /// @brief STAGED raw restore, stage 1 of 3 (see the header contract):
    ///        the empty-arena contract, the byte-bump span, and the bulk
    ///        page-block acquisition.
    ///
    /// @param vidCount       The vid high-water the chunks will reproduce.
    /// @param liveCount      Live pages to carve.
    /// @param byteBumpCursor The byte-bump span to reproduce (expected 0).
    void LbArena::restoreForRawLoadBegin(int32_t vidCount, int32_t liveCount,
                                         ArenaOffset byteBumpCursor) {
        assert(resident_ && "restoreForRawLoadBegin on a deloaded arena");
        // Restore on a non-empty arena is a bug (Rule 19) — a raw reload is only
        // ever the FIRST population of a freshly-marked-resident arena.
        assert(blocks_.empty() && pageBlocks_.empty() && pageTable_.empty()
            && freeHead_ == nullptr && freePageCount_ == 0
            && freeVidHead_ == -1
            && livePages_ == 0 && cursor_ == 0
            && "restoreForRawLoadBegin on a non-empty arena");
        assert(vidCount >= 0 && liveCount >= 0 && liveCount <= vidCount);
        ensureGeometry();
        const int32_t pb = global_->pageBytes();
        const int32_t ppb = blockBytes_ / pb;

        // Byte-bump tier (usually absent: byteBumpCursor == 0 in production).
        if (byteBumpCursor != 0) {
            const int64_t needBlocks =
                (static_cast<int64_t>(byteBumpCursor) + blockBytes_ - 1)
                >> blockShift_;
            acquireInto(blocks_, static_cast<int32_t>(needBlocks));
            cursor_ = byteBumpCursor;
        }

        const int32_t keptBlocks = (liveCount + ppb - 1) / ppb;   // 0 if none
        if (keptBlocks > 0) acquireInto(pageBlocks_, keptBlocks);
        // The final live-page count; the chunks' running `slot` must land
        // exactly here (asserted in restoreForRawLoadEnd). Nothing reads
        // livePages_ between the stages — assertInvariants runs only at End.
        livePages_ = liveCount;
    }

    /// @brief STAGED raw restore, stage 2 of 3 (see the header contract):
    ///        bind one bitmap chunk's vids — live vids to the next dense
    ///        slots, dead vids to interior holes.
    ///
    /// @param bits     `ceil(vidSpan / 8)` bitmap bytes for this chunk.
    /// @param startVid First vid of the chunk; a multiple of 8; must equal
    ///                 the number of vids already bound.
    /// @param vidSpan  Vids in this chunk; >= 0.
    /// @param slot     Dense slots filled so far (live vids bound).
    /// @return The updated dense-slot count after this chunk.
    int32_t LbArena::restoreForRawLoadChunk(const unsigned char* bits,
                                            int32_t startVid, int32_t vidSpan,
                                            int32_t slot) {
        assert(resident_ && "restoreForRawLoadChunk on a deloaded arena");
        assert((startVid & 7) == 0
            && "restoreForRawLoadChunk: chunk must start on a byte boundary");
        assert(startVid == pageTable_.size()
            && "restoreForRawLoadChunk: chunks must arrive in ascending, "
               "gapless vid order");
        assert(vidSpan >= 0 && slot >= 0 && slot <= startVid);
        const int32_t pb = global_->pageBytes();
        const int32_t ppb = blockBytes_ / pb;
        for (int32_t i = 0; i < vidSpan; ++i) {
            const bool live = (bits[i >> 3]
                & static_cast<unsigned char>(1u << (i & 7))) != 0;
            if (live) {
                char* page = pageBlocks_[slot / ppb]
                    + static_cast<std::ptrdiff_t>(slot % ppb) * pb;
                pageTable_.push_back(page);
                ++slot;
            } else {
                // Dead vid: a chain-node placeholder; restoreForRawLoadEnd
                // rebuilds the canonical free-vid chain over all dead slots.
                pageTable_.push_back(deadSlotEncode(-1));
            }
        }
        return slot;
    }

    /// @brief STAGED raw restore, stage 3 of 3 (see the header contract):
    ///        the free-list rebuild + the shape proof.
    ///
    /// @param vidCount The vid high-water stage 1 announced.
    /// @param slot     The final dense-slot count from the last chunk.
    void LbArena::restoreForRawLoadEnd(int32_t vidCount, int32_t slot) {
        assert(resident_ && "restoreForRawLoadEnd on a deloaded arena");
        assert(pageTable_.size() == vidCount
            && "restoreForRawLoadEnd: chunks did not cover exactly vidCount "
               "vids");
        assert(slot == livePages_
            && "restoreForRawLoadEnd: bitmap set-bit count != liveCount");
        // Rebuild the free-vid chain canonically from the bitmap-derived dead
        // set: walk vids DESCENDING pushing each dead vid, so the LIFO head is
        // the LOWEST dead vid and pops ascend — a pure function of the live
        // bitmap (deterministic; the pre-dump LIFO history is not serialized,
        // deliberately: vid values never reach the canonical proof bytes, and
        // the canonical rebuild keeps reload behavior a function of content).
        freeVidHead_ = -1;
        for (int32_t v = vidCount; v-- > 0; ) {
            if (!isLivePageSlot(pageTable_.peek(v))) {
                pageTable_.set(v, deadSlotEncode(freeVidHead_));
                freeVidHead_ = v;
            }
        }
        const int32_t pb = global_->pageBytes();
        const int32_t ppb = blockBytes_ / pb;
        const int32_t keptBlocks = (livePages_ + ppb - 1) / ppb;
        // Rebuild the intrusive free-list from the tail slots of the last kept
        // block (ascending) — deterministic from the live count alone, the
        // reverse of compactPages' rebuild.
        freeHead_ = nullptr;
        freePageCount_ = 0;
        for (int32_t s = livePages_; s < keptBlocks * ppb; ++s) {
            char* page = pageBlocks_[s / ppb]
                + static_cast<std::ptrdiff_t>(s % ppb) * pb;
            *reinterpret_cast<char**>(page + pb - sizeof(char*)) = freeHead_;
            freeHead_ = page;
            ++freePageCount_;
        }
        assertInvariants();
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
            && freeVidHead_ == -1
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
        releaseFrom(blocks_);    // bulk return the byte-bump data blocks
        releaseFrom(pageBlocks_);  // bulk return the page-tier data blocks
        blocks_.clear();         // each clear() also returns that table's own
        pageBlocks_.clear();     // spilled directory blocks
        pageTable_.clear();
        freeHead_ = nullptr;     // the free pages lived in pageBlocks_ (returned)
        freePageCount_ = 0;
        freeVidHead_ = -1;       // the chain nodes lived in pageTable_ (cleared)
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
        // livePages_ equals the live page-table entries; every slot is either
        // a live page or a tagged free-vid chain node (never plain null); the
        // free-vid chain visits exactly the dead slots and terminates.
        const int32_t hw = pageTable_.size();
        int32_t live = 0;
        for (int32_t v = 0; v < hw; ++v) {
            assert(pageTable_[v] != nullptr
                && "LbArena::assertInvariants: null page-table slot — dead "
                   "vids must be free-vid chain nodes");
            if (isLivePageSlot(pageTable_[v])) ++live;
        }
        assert(live == livePages_
            && "LbArena::assertInvariants: livePages_ != live page count");
        {
            // Walk the free-vid chain exactly (hw - live) steps: each node is
            // an in-range dead slot; the walk must end at -1 precisely then —
            // proves length, termination, and no cycle.
            const int32_t deadCount = hw - live;
            int32_t v = freeVidHead_;
            for (int32_t step = 0; step < deadCount; ++step) {
                assert(v >= 0 && v < hw
                    && "LbArena::assertInvariants: free-vid chain node out of "
                       "range");
                assert(!isLivePageSlot(pageTable_[v])
                    && "LbArena::assertInvariants: free-vid chain node is a "
                       "live vid");
                v = deadSlotNextVid(pageTable_[v]);
            }
            assert(v == -1
                && "LbArena::assertInvariants: free-vid chain length != dead "
                   "slot count");
        }
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
            if (!isLivePageSlot(page)) continue;
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
