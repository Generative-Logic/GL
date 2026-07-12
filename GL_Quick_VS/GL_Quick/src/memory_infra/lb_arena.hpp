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
#include "ptr_directory.hpp"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace gl {

    /// @brief A virtual byte offset into an `LbArena`, relative to the arena
    ///        base — the bump model's "pointer".
    ///
    /// @details
    /// An offset is INTERPRETED to a physical address at every dereference
    /// (`LbArena::resolve`): it carries no physical address, so a dumped
    /// arena image relocates onto an entirely different set of physical
    /// blocks with zero fixup, and the value itself is deterministic (a pure
    /// function of allocation order, which the prover drives single-threaded
    /// per LB). Offsets are the storage form of every cold-path link;
    /// physical addresses are computed on demand and never stored.
    using ArenaOffset = uint32_t;

    /// @brief Sentinel "no offset", distinct from every real allocation
    ///        (offset 0 is the first valid allocation, so 0 cannot serve as
    ///        a null).
    constexpr ArenaOffset kNullOffset = 0xFFFFFFFFu;

    /// @brief Byte written over popped (tail-freed) and rewound / reset arena
    ///        spans so a stale read of dead memory announces itself as
    ///        recognisable garbage.
    constexpr unsigned char kArenaPoisonByte = 0xCD;

    /// @brief The shared bump arena over pool blocks, addressed by virtual
    ///        offsets — one management code backs the per-LB store, the
    ///        per-worker scratch arenas, and the sealed-page handoff (all draw
    ///        the cold `acquireBlock` grant path; they differ in lifetime, not
    ///        grant path).
    ///
    /// @details
    /// It draws `static_block_bytes` blocks from the `GlobalMemoryManager` and
    /// lays every container's bytes into ONE virtually-contiguous space: block
    /// 0 occupies virtual `[0, blockBytes)`, block 1 `[blockBytes,
    /// 2*blockBytes)`, and so on. `alloc` bumps a virtual cursor by the request
    /// size; `resolve` splits an offset into `(blockIndex, withinBlock)` and
    /// returns the physical address. The block size is a power of two, so the
    /// split is one shift plus one mask. The per-LB store rides the byte bump +
    /// the page tier and round-trips to SSD (deload) and compacts; the
    /// per-worker scratch / sealed arenas ride the BYTE-BUMP tier (`allocBytes`
    /// = a resolved `alloc`, with `mark` / `rewind` / `reset` over the cursor)
    /// and are released per task, never deloaded or compacted, so a resolved
    /// scratch pointer stays valid until its rewind / releaseAll. `allocBytes`
    /// is the byte tier, never the page tier — the page tier is exclusively
    /// `allocPage` containers, so scratch fill and paged containers never
    /// collide on one arena.
    ///
    /// An allocation never straddles a block (the cursor pads to the next
    /// block boundary when a request will not fit the current block's
    /// remainder), so every allocation is one contiguous object and `resolve`
    /// returns a single pointer. `alloc` aligns the cursor to the requested
    /// alignment first; block bases are pool-aligned (`::operator new`, at
    /// least 16) and the block size is a whole multiple of any supported
    /// alignment, so the resolved address inherits the requested alignment.
    ///
    /// `popTo` is the only byte-bump free path: it rolls the cursor back to a
    /// captured position (last-in-first-out) and poisons the reclaimed span.
    /// Interior dead space left by a non-tail erase is NOT reclaimed here; it
    /// becomes a hole that the copying compaction collects (a later batch).
    /// Blocks are retained across `popTo` (reused by the next `alloc`); they
    /// return to the pool only at `releaseAll` and at compaction.
    ///
    /// NO lock: per-LB state is mutated single-threaded by the prover's
    /// phase model (the parallel phase only reads cold containers). The one
    /// cross-thread touch is the global manager's own mutex inside
    /// `acquireBlock` / `releaseBlock`.
    ///
    /// @invariant No observable behavior depends on which physical block an
    ///            offset resolves to; offsets are deterministic and the
    ///            block grant order from the global manager is absorbed by
    ///            resolution (the bump form of
    ///            I-107).
    /// @invariant Pool exhaustion (a fresh block is needed and none remains)
    ///            is an assert naming `static_pool_bytes`, never a fallback —
    ///            it surfaces from `GlobalMemoryManager::acquireBlock`.
    /// @see `GlobalMemoryManager` (block source), `ScratchArena` (the
    ///      per-worker alias of this class).
    class LbArena {
    public:
        /// @brief Construct an UNBOUND arena (no pool, no geometry) — bound
        ///        later by `bind`.
        ///
        /// @details
        /// The scratch registry default-constructs its per-slot arenas, then
        /// binds each with `bind` once the pool is known (a slot must consume
        /// nothing until first use). A per-LB arena uses the global-taking
        /// constructor instead and is bound immediately.
        LbArena() = default;

        /// @brief Bind to the global manager that grants this arena's blocks.
        ///
        /// @details
        /// Lazy: holds zero blocks and computes no geometry until the first
        /// `alloc` — transient LB objects must not consume pool blocks (or
        /// touch the not-yet-initialized pool) merely by existing.
        ///
        /// @param global The block source (process-wide in production; a
        ///               private instance in unit tests). Must be non-null.
        explicit LbArena(GlobalMemoryManager* global);

        /// @brief Returns every held block to the global manager.
        ~LbArena();

        LbArena(const LbArena&) = delete;
        LbArena& operator=(const LbArena&) = delete;
        LbArena(LbArena&&) = delete;
        LbArena& operator=(LbArena&&) = delete;

        /// @brief Lazily bind a default-constructed arena to a pool.
        ///
        /// @details Sets the block source on a default arena, leaving geometry
        /// and blocks lazy (acquired on first `alloc`). Used where an arena
        /// cannot bind at construction because its pool is initialized later —
        /// the routing mailboxes on the mail pool (the root LB predates the
        /// pool init), and the per-worker scratch arenas in their registry.
        /// Asserts the arena is unbound and the pool is initialized.
        ///
        /// @param global Pool to draw blocks from.
        void bind(GlobalMemoryManager* global);

        /// @brief Bump-allocate `bytes`, aligned to `align`, and return the
        ///        new allocation's virtual offset.
        ///
        /// @details
        /// Aligns the cursor up to `align`, pads to the next block boundary
        /// if the request would straddle the current block, acquires fresh
        /// blocks from the global manager as the cursor crosses block
        /// boundaries, then advances the cursor by `bytes`. Asserts
        /// residency, a positive length that fits one block, and a
        /// power-of-two `align` no greater than the block alignment.
        ///
        /// @param bytes Exact length of the allocation; `0 < bytes <=
        ///              blockBytes`.
        /// @param align Power-of-two alignment of the returned offset, in
        ///              `[1, 16]` (the maximum element alignment the cold
        ///              containers carry). Defaults to 1 (byte strings).
        /// @return The virtual offset of the allocation; resolve it for the
        ///         physical address.
        ArenaOffset alloc(int32_t bytes, int32_t align = 1);

        /// @brief Resolve a virtual offset to its physical address
        ///        (mutable).
        ///
        /// @details
        /// The hot path: one shift, one mask, one block-table lookup. The
        /// returned pointer is valid only while the arena is resident and
        /// only until the next operation that can move bytes (compaction,
        /// `releaseAll`); it must never be stored across such an operation
        /// (I-107). Asserts residency and
        /// that `off` lies inside the allocated region.
        ///
        /// @param off A virtual offset previously returned by `alloc` (or
        ///            derived from one), `< cursor()`.
        /// @return Pointer into block storage at `off`.
        char* resolve(ArenaOffset off) {
            assert(resident_ && "LbArena::resolve on a deloaded arena");
            assert(blockBytes_ != 0 && "LbArena::resolve before any alloc");
            assert(off < cursor_
                && "LbArena::resolve of an unallocated offset");
            const int64_t block = static_cast<int64_t>(off) >> blockShift_;
            assert(block < static_cast<int64_t>(blocks_.size()));
            return blocks_.peek(static_cast<int32_t>(block))
                 + (off & blockMask_);
        }

        /// @brief Resolve a virtual offset to its physical address
        ///        (read-only).
        ///
        /// @param off A virtual offset `< cursor()`.
        /// @return Const pointer into block storage at `off`.
        const char* resolve(ArenaOffset off) const {
            assert(resident_ && "LbArena::resolve on a deloaded arena");
            assert(blockBytes_ != 0 && "LbArena::resolve before any alloc");
            assert(off < cursor_
                && "LbArena::resolve of an unallocated offset");
            const int64_t block = static_cast<int64_t>(off) >> blockShift_;
            assert(block < static_cast<int64_t>(blocks_.size()));
            return blocks_.peek(static_cast<int32_t>(block))
                 + (off & blockMask_);
        }

        /// @brief The current bump position — the offset of the next byte the
        ///        arena would hand out (before alignment / straddle padding).
        ///
        /// @return The virtual cursor; capture it before an allocation to
        ///         later `popTo` that point.
        ArenaOffset cursor() const { return cursor_; }

        /// @brief Tail-free back to a previously captured cursor position.
        ///
        /// @details
        /// Rolls the cursor back to `off` (last-in-first-out) and poisons
        /// the reclaimed span with `kArenaPoisonByte`. Retained blocks stay
        /// retained — the next `alloc` reuses them. Legal only backwards
        /// (`off <= cursor()`); a no-op when `off == cursor()`.
        ///
        /// @param off A cursor value captured by an earlier `cursor()`.
        void popTo(ArenaOffset off);

        /// @brief Whether the arena's blocks are in memory (not deloaded).
        ///
        /// @return `true` between construction / `markResident` and
        ///         `markDeloaded`.
        bool resident() const { return resident_; }

        /// @brief Flag the arena as deloaded to SSD.
        ///
        /// @details
        /// Asserts `releaseAll` already ran (no blocks held) — the flag
        /// records a completed deload, it does not perform one.
        void markDeloaded();

        /// @brief Flag the arena as resident again (reload completed).
        void markResident();

        /// @brief Return every held block to the global manager and reset the
        ///        cursor to empty (cached geometry is kept).
        ///
        /// @details
        /// Legal at deload or teardown, when every container's offsets are
        /// simultaneously rebuilt (reload) or discarded (destruction). After
        /// this, the cursor restarts at 0.
        void releaseAll();

        /// @brief Blocks currently held (telemetry / tests) — byte-bump and
        ///        page-tier blocks together.
        ///
        /// @details
        /// "Held" sums every pool block the LB pins: the byte-bump storage blocks
        /// (`blocks_`'s entries), the page-tier carved blocks (`pageBlocks_`'s
        /// entries), and each of the three `PtrDirectory` tables' own spilled
        /// directory blocks (`blocks_` / `pageBlocks_` / `pageTable_` — non-zero
        /// only once a table outgrows its inline buffer). This is what the
        /// reshuffle and eviction accounting reason about; counting only one
        /// substrate would under-count an LB whose statements are paged and whose
        /// strings are byte-bumped.
        ///
        /// @return Total blocks granted to this arena and not returned.
        int64_t blocksHeld() const {
            return static_cast<int64_t>(blocks_.size() + pageBlocks_.size())
                 + blocks_.blocksHeld() + pageBlocks_.blocksHeld()
                 + pageTable_.blocksHeld();
        }

        /// @brief Current used span as a byte measure (telemetry /
        ///        scratch-string liveness).
        ///
        /// @details
        /// The byte-bump cursor. A scratch / sealed arena fills its bytes
        /// through `allocBytes` (a resolved byte-bump `alloc`), so the cursor is
        /// its used span; a per-LB container arena's paged storage is not
        /// counted here (the page tier carries no cursor). The scratch string
        /// views read it as the birth position for their rewind tripwire.
        ///
        /// @return The arena's used-byte span.
        int64_t usedBytes() const {
            return static_cast<int64_t>(cursor_);
        }

        /// @brief Bytes per block, forwarded from the global manager.
        ///
        /// @details
        /// The geometry constant the deload header records as a sanity field
        /// and that bounds a single allocation. Asserts the pool is
        /// initialized (lock-free, immutable after init).
        ///
        /// @return `static_block_bytes` of the hierarchy config.
        int32_t blockBytes() const { return global_->blockBytes(); }

        /// @brief Bytes per page (the page-tier allocation unit), forwarded
        ///        from the global manager.
        ///
        /// @details
        /// The geometry constant a paged container splits into its
        /// elements-per-page (`pageBytes / sizeof(T)`, rounded down to a power
        /// of two) and its directory fan-out (`pageBytes / sizeof(vid)`).
        ///
        /// @return `static_page_bytes` of the hierarchy config.
        int32_t pageBytes() const { return global_->pageBytes(); }

        /// @brief Forward a paged container's single -> two-level page-directory
        ///        promotion to the manager's telemetry.
        ///
        /// @details
        /// `PagedVector` / `PagedHashIndex` call this once when their directory
        /// crosses `dirCap` data pages. Pure observability (Rule 16 / I-44) —
        /// the manager only counts it; nothing in the prover reads it back.
        ///
        /// @param numPages Data pages the container holds at the promotion.
        void recordTwoLevelPromotion(int32_t numPages) {
            global_->recordTwoLevelPromotion(numPages);
        }

        // ---- Page tier (the per-LB paged-container substrate) ------------
        // Page-granular allocation alongside the byte bump: a container that
        // wants contiguous storage takes whole pages (allocPage) and packs its
        // elements inside them, returning pages (freePage) as it shrinks. The
        // page tier draws its own blocks from the same pool (mode-aware) and
        // carves each into static_page_bytes pages. Pages are addressed by a
        // stable virtual id (vid) resolved through pageAt, so the background
        // compaction can rebind a vid to a different physical page without any
        // container noticing (I-107). The byte
        // bump above is retired once every container is paged.

        /// @brief Allocate one page and return its virtual id (vid).
        ///
        /// @details
        /// Reuses a page parked on the free-list when one is available;
        /// otherwise carves the next page, acquiring a fresh block
        /// (`acquireBlock`) and carving it
        /// into `pageBytes` pages the first time the free-list runs dry. The
        /// returned vid is the next slot of the page table; vids roll back when
        /// the tail page is freed (`freePage`), so the id space stays compact
        /// under stack-like (hot per-scope) use. Asserts residency.
        ///
        /// @return The new page's vid — a stable handle until the page is
        ///         freed.
        /// @invariant `pageAt(vid)` resolves to `pageBytes()` writable bytes
        ///            until `freePage(vid)`.
        int32_t allocPage();

        /// @brief Free the page at `vid`, returning it to the free-list.
        ///
        /// @details
        /// Poisons the page (`poisonByte_`) so a stale read announces itself,
        /// parks the physical page on the free-list for reuse, and marks the
        /// vid dead. Freeing the CURRENT tail vid (and any freed vids exposed
        /// beneath it) rolls the page cursor back — the page-level bump-back
        /// the hot per-call / per-scope windows rely on; freeing an interior
        /// vid leaves a hole the free-list fills later. Asserts residency and
        /// that the vid is live.
        ///
        /// @param vid A vid returned by `allocPage` and not yet freed.
        void freePage(int32_t vid);

        /// @brief Resolve a live vid to its physical page (mutable).
        ///
        /// @details
        /// One page-table lookup. The returned pointer is valid only while the
        /// arena is resident and only until an operation that can move the page
        /// (the background compaction, `releaseAll`); it must never be stored
        /// across such an operation (I-107).
        /// Residency and vid liveness are preconditions; the per-resolve asserts
        /// that checked them are gated off (`#if 0`) on this hottest path for RT.
        ///
        /// @param vid A live vid.
        /// @return Pointer to `pageBytes()` bytes of page storage.
        char* pageAt(int32_t vid) {
            // Per-resolve asserts gated under GL_ARENA_PARANOID on this hottest
            // path for RT (the unit tests and the paranoid run arm them).
#if GL_ARENA_PARANOID
            assert(resident_ && "LbArena::pageAt on a deloaded arena");
            assert(vid >= 0 && vid < pageTable_.size()
                && "LbArena::pageAt on an out-of-range vid");
            assert(pageTable_.peek(vid) != nullptr
                && "LbArena::pageAt on a freed vid");
#endif
            return pageTable_.peek(vid);
        }

        /// @brief Resolve a live vid to its physical page (read-only).
        ///
        /// @param vid A live vid.
        /// @return Const pointer to `pageBytes()` bytes of page storage.
        const char* pageAt(int32_t vid) const {
            // Per-resolve asserts gated under GL_ARENA_PARANOID on this hottest
            // path for RT (the unit tests and the paranoid run arm them).
#if GL_ARENA_PARANOID
            assert(resident_ && "LbArena::pageAt on a deloaded arena");
            assert(vid >= 0 && vid < pageTable_.size()
                && "LbArena::pageAt on an out-of-range vid");
            assert(pageTable_.peek(vid) != nullptr
                && "LbArena::pageAt on a freed vid");
#endif
            return pageTable_.peek(vid);
        }

        /// @brief Pages currently live (allocated and not yet freed).
        ///
        /// @return Live page count (telemetry / tests).
        int32_t livePages() const { return livePages_; }

        /// @brief Page id high-water — the page cursor, i.e. the next vid
        ///        `allocPage` would issue (equals the page-table size).
        ///
        /// @details
        /// The hot per-scope windows capture it as a mark and free back down to
        /// it; it shrinks on tail `freePage`.
        ///
        /// @return The current vid count.
        int32_t pageHighWater() const {
            return pageTable_.size();
        }

        /// @brief Physical pages parked on the free-list for reuse
        ///        (telemetry / tests).
        ///
        /// @return Free-list length.
        int32_t freePageCount() const {
            return freePageCount_;
        }

        /// @brief In-place page compaction — pack the live pages onto the
        ///        contiguous prefix of the held blocks and return the emptied
        ///        blocks to the pool. The background steward's reclaim.
        ///
        /// @details
        /// Content-invisible: only vid→physical bindings move — containers hold
        /// stable vids and resolve through `pageAt`, so none needs touching or
        /// a pointer-cache refresh. Runs under exclusive LB access (steward
        /// claim / single-threaded barrier), never mid-burst
        /// (I-107). See the definition for the
        /// permutation mechanics (chain-walk + single-scratch-page cycles).
        ///
        /// The permutation scratch (live-vid list, rank/slot maps, path/cycle
        /// walks, the one rotate page) lives on the caller-supplied `scr`, drawn
        /// from a pool INDEPENDENT of the deloadable one being compacted — the
        /// reclaim must never depend on the pool it frees. `scr` must be exclusive
        /// to the calling thread (the discharge path and the steward each own one,
        /// since they can run concurrently).
        ///
        /// @param scr A scratch arena bound to a never-deloaded pool (e.g.
        ///            `lbMemory()`), exclusive to this thread; returned with no
        ///            live pages (its blocks retained for reuse).
        /// @return Number of blocks returned to the global pool.
        int64_t compactPages(LbArena& scr);

        // ---- Raw arena image (the near-memcpy eviction/reload datapath) -----
        // The page tier IS the entire deloadable state (every cold container
        // packs its elements into `allocPage` pages of this one arena, and the
        // throw-away `PagedHashIndex` bucket arrays ride the same page tier). So
        // a byte image of the live pages, in ascending vid order, plus a bitmap
        // of which vids are live, restores the arena to a byte-identical logical
        // state with ZERO container work: vids are position-independent handles
        // (I-107), so the reload binds the same vids to freshly carved DENSE
        // pages and every container's scalar bookkeeping (unchanged in the
        // never-deloaded `Memory` shell) still resolves. No element walk, no
        // per-key index rebuild. Bytes are NONDETERMINISTIC (I-103 is waived for
        // eviction images, user-approved) but the restored LOGICAL state is
        // byte-identical, so proof output stays deterministic.

        /// @brief The scalar shape a raw image records in its header — enough
        ///        for `restoreForRawLoad` to rebuild the page tier.
        ///
        /// @details
        /// `byteBumpCursor` is the byte-bump tier's used span (`cursor_`),
        /// EXPECTED 0 for a deloadable LB arena (every deloadable container is
        /// page-tier — no `ArenaVector` members, no direct `alloc`/`allocBytes`
        /// on the manager); the v4 format carries a nonzero span raw so a future
        /// byte-bump container is supported. `vidCount` is the vid high-water
        /// (`pageTable_.size()`, including any interior dead-vid holes);
        /// `livePages` is the number of set bits the live bitmap will carry.
        struct RawShape {
            int32_t blockBytes;         // grant unit (header sanity field)
            int32_t pageBytes;          // page unit (header sanity field)
            ArenaOffset byteBumpCursor; // cursor_ (expected 0 in production)
            int32_t vidCount;           // pageTable_.size() (holes included)
            int32_t livePages;          // live (non-null) vids
        };

        /// @brief The arena's raw-image shape — the header scalars.
        ///
        /// @details
        /// Pure read (telemetry-cheap): reads the cached geometry and the page-
        /// table high-water / live count. Legal on an empty arena (returns
        /// zeros for the tier fields), which is the header-only raw image an
        /// empty aggregate writes. Asserts residency — a deloaded arena has no
        /// shape to report.
        ///
        /// @return The `RawShape` the raw dumper stamps into the v4 header.
        RawShape rawShape() const;

        /// @brief Fill `bitmap` with one bit per vid — set iff the vid is live.
        ///
        /// @details
        /// The raw header's live-vid bitmap: bit `v` (byte `v/8`, bit `v%8`) is
        /// set iff `pageTable_[v]` is a live page. Interior holes (dead vids
        /// freed out of tail order) read 0 and are preserved across the round
        /// trip, so the reload rebinds exactly the vids that were live. Reads
        /// only the page table; asserts residency and an exactly-sized buffer.
        ///
        /// @param bitmap      Caller buffer of `bitmapBytes` bytes, zeroed then
        ///                    filled.
        /// @param bitmapBytes Must equal `ceil(vidCount / 8)`.
        void fillLiveBitmap(unsigned char* bitmap, int32_t bitmapBytes) const;

        /// @brief Fill a bit-chunk of the live-vid bitmap for vids
        ///        `[startVid, startVid + vidSpan)` — the dump-side chunk
        ///        primitive behind the dynamic v4 header.
        ///
        /// @details
        /// Same bit semantics as `fillLiveBitmap` (bit `i` of the chunk =
        /// vid `startVid + i` live), restricted to a caller-chosen range so
        /// the raw dumper can stream an arbitrarily large bitmap through one
        /// bounded stack buffer instead of the single fixed-size buffer the
        /// old 4 KiB header capped. `startVid` must be a multiple of 8
        /// (chunks split on byte boundaries, so the chunk bytes concatenate
        /// to the exact whole-range bitmap). `fillLiveBitmap` is the
        /// whole-range convenience wrapper over this.
        ///
        /// @param bitmap   Caller buffer of `ceil(vidSpan / 8)` bytes, zeroed
        ///                 then filled.
        /// @param startVid First vid of the chunk; a multiple of 8; >= 0.
        /// @param vidSpan  Vids in the chunk; `startVid + vidSpan` must not
        ///                 exceed the vid high-water; >= 0.
        void fillLiveBitmapRange(unsigned char* bitmap, int32_t startVid,
                                 int32_t vidSpan) const;

        /// @brief Stream the arena's raw payload to a byte sink — the byte-bump
        ///        span (usually absent) then the live pages, ascending vid.
        ///
        /// @details
        /// Heap-free: walks live pages in ascending vid order and COALESCES
        /// physically-adjacent pages into one `sink` call (fewer file writes),
        /// which is invisible to the byte stream — the payload is always
        /// `livePages * pageBytes` bytes in ascending-vid order regardless of
        /// how the source pages are laid out. The byte-bump span (if
        /// `cursor_ != 0`) is emitted FIRST, matching the v4 header order.
        /// Read-only; asserts residency. The `Sink` is any callable
        /// `void(const char* data, int64_t len)`.
        ///
        /// @tparam Sink  A callable `void(const char*, int64_t)`.
        /// @param  sink  Receives each contiguous run of payload bytes.
        template <class Sink>
        void emitRawImage(Sink&& sink) const {
            assert(resident_ && "emitRawImage on a deloaded arena");
            // Byte-bump span first (usually absent: cursor_ == 0 in production).
            if (cursor_ != 0) {
                ArenaOffset p = 0;
                while (p < cursor_) {
                    const int64_t block =
                        static_cast<int64_t>(p) >> blockShift_;
                    const int64_t blockEnd =
                        (block + 1) << static_cast<int64_t>(blockShift_);
                    const ArenaOffset spanEnd =
                        (static_cast<int64_t>(cursor_) < blockEnd)
                            ? cursor_ : static_cast<ArenaOffset>(blockEnd);
                    sink(blocks_.peek(static_cast<int32_t>(block))
                             + (p & blockMask_),
                         static_cast<int64_t>(spanEnd - p));
                    p = spanEnd;
                }
            }
            // Live pages, ascending vid, coalescing physically-adjacent runs.
            const int32_t hw = pageTable_.size();
            const int32_t pb = global_->pageBytes();
            const char* runStart = nullptr;
            int64_t runLen = 0;
            for (int32_t v = 0; v < hw; ++v) {
                const char* page = pageTable_.peek(v);
                if (page == nullptr) continue;   // interior dead-vid hole
                if (runStart != nullptr && page == runStart + runLen) {
                    runLen += pb;
                } else {
                    if (runStart != nullptr) sink(runStart, runLen);
                    runStart = page;
                    runLen = pb;
                }
            }
            if (runStart != nullptr) sink(runStart, runLen);
        }

        /// @brief Rebuild a FRESH page tier that binds ascending live vids to
        ///        consecutive dense pages — the raw reload's arena side.
        ///
        /// @details
        /// Bulk-acquires the blocks the byte-bump span and the live pages need
        /// (one mutex batch, `GlobalMemoryManager::acquireBlocks`), binds the
        /// k-th ascending live vid to slot `k` (dense — the reload self-heals
        /// fragmentation), nulls the dead vids so interior holes survive, and
        /// rebuilds the free-list from the tail slots. Leaves the pages
        /// UNINITIALISED — the caller fills them via `fillRawImage`. Because the
        /// vids are preserved and pages resolve through `pageAt`, every
        /// container's scalar state (`size_`/`rootVid_`/`numPages_`, in the
        /// never-deloaded `Memory` shell) still resolves — no container touch,
        /// no index rebuild. Asserts the arena is EMPTY (restore on a non-empty
        /// arena is a bug, Rule 19) and residency (the caller `markResident`s
        /// first). `assertInvariants` at the end proves the rebuilt shape.
        ///
        /// @param vidCount       The vid high-water to reproduce (holes
        ///                       included); >= 0.
        /// @param liveBitmap     `ceil(vidCount/8)` bytes; bit `v` set iff vid
        ///                       `v` is live.
        /// @param liveCount      Number of set bits in `liveBitmap`; the pages
        ///                       to carve; `0 <= liveCount <= vidCount`.
        /// @param byteBumpCursor The byte-bump span to reproduce (expected 0).
        void restoreForRawLoad(int32_t vidCount,
                               const unsigned char* liveBitmap,
                               int32_t liveCount, ArenaOffset byteBumpCursor);

        /// @brief STAGED raw restore, stage 1 of 3: geometry, the byte-bump
        ///        span, and the bulk block acquisition — the chunked twin of
        ///        `restoreForRawLoad`'s entry, for a bitmap that streams
        ///        through a bounded buffer (the dynamic v4 header).
        ///
        /// @details
        /// The loader reads an arbitrarily large live-vid bitmap in bounded
        /// 4 KiB chunks straight off the file, so the restore is split:
        /// `restoreForRawLoadBegin` (this) validates the empty-arena
        /// contract, reproduces the byte-bump span, bulk-acquires the kept
        /// page blocks, and records `liveCount` as the live-page count;
        /// `restoreForRawLoadChunk` binds each bitmap chunk's vids in
        /// ascending order; `restoreForRawLoadEnd` rebuilds the free-list
        /// and proves the shape. `restoreForRawLoad` is the single-shot
        /// wrapper (one chunk covering every vid) — same asserts, same
        /// result, byte-for-byte.
        ///
        /// @param vidCount       The vid high-water the chunks will
        ///                       reproduce; >= 0.
        /// @param liveCount      Live pages to carve; `0 <= liveCount <=
        ///                       vidCount`.
        /// @param byteBumpCursor The byte-bump span to reproduce (expected
        ///                       0).
        void restoreForRawLoadBegin(int32_t vidCount, int32_t liveCount,
                                    ArenaOffset byteBumpCursor);

        /// @brief STAGED raw restore, stage 2 of 3: bind the vids of one
        ///        bitmap chunk — bit `i` = vid `startVid + i`, live vids to
        ///        the next dense slots, dead vids to interior holes.
        ///
        /// @details
        /// Chunks must arrive in ascending, gapless order (`startVid` equals
        /// the vids bound so far — asserted) and split on byte boundaries
        /// (`startVid` a multiple of 8, mirroring
        /// `fillLiveBitmapRange`). The running dense-slot cursor travels as
        /// the caller's `slot` (in) / return value (out), so the arena keeps
        /// no cross-stage scratch state.
        ///
        /// @param bits     `ceil(vidSpan / 8)` bitmap bytes for this chunk.
        /// @param startVid First vid of the chunk; a multiple of 8; must
        ///                 equal the number of vids already bound.
        /// @param vidSpan  Vids in this chunk; >= 0.
        /// @param slot     Dense slots filled so far (live vids bound).
        /// @return The updated dense-slot count after this chunk.
        int32_t restoreForRawLoadChunk(const unsigned char* bits,
                                       int32_t startVid, int32_t vidSpan,
                                       int32_t slot);

        /// @brief STAGED raw restore, stage 3 of 3: rebuild the free-list
        ///        from the tail slots and prove the rebuilt shape.
        ///
        /// @details
        /// Asserts the chunks covered exactly `vidCount` vids, that the
        /// bound live count matches stage 1's `liveCount` (`slot`), and the
        /// dump-time last-vid-live invariant (the page-table tail is never a
        /// freed vid), then `assertInvariants`.
        ///
        /// @param vidCount The vid high-water stage 1 announced.
        /// @param slot     The final dense-slot count from the last chunk.
        void restoreForRawLoadEnd(int32_t vidCount, int32_t slot);

        /// @brief Fill the pages `restoreForRawLoad` carved with raw bytes from
        ///        a source — the heap-free read-into-pages mechanism.
        ///
        /// @details
        /// The mirror of `emitRawImage`: walks the byte-bump span (if any) then
        /// the dense live pages in ascending vid = ascending slot order,
        /// coalescing physically-adjacent slots into one `fill` call, and hands
        /// each contiguous DEST run to the source. The source reads the file
        /// bytes straight into pool memory (one copy total, no staging buffer).
        /// The run boundaries need not match `emitRawImage`'s — both walk the
        /// same total byte sequence in ascending-vid order, so the flat file
        /// payload lands byte-for-byte where it belongs. Asserts residency. The
        /// `Fill` is any callable `void(char* dst, int64_t len)`.
        ///
        /// @tparam Fill A callable `void(char*, int64_t)` filling `len` bytes at
        ///              `dst`.
        /// @param  fill Fills each contiguous run of the restored pages.
        template <class Fill>
        void fillRawImage(Fill&& fill) {
            assert(resident_ && "fillRawImage on a deloaded arena");
            // Byte-bump span first (matches emitRawImage / the header order).
            if (cursor_ != 0) {
                ArenaOffset p = 0;
                while (p < cursor_) {
                    const int64_t block =
                        static_cast<int64_t>(p) >> blockShift_;
                    const int64_t blockEnd =
                        (block + 1) << static_cast<int64_t>(blockShift_);
                    const ArenaOffset spanEnd =
                        (static_cast<int64_t>(cursor_) < blockEnd)
                            ? cursor_ : static_cast<ArenaOffset>(blockEnd);
                    fill(blocks_.peek(static_cast<int32_t>(block))
                             + (p & blockMask_),
                         static_cast<int64_t>(spanEnd - p));
                    p = spanEnd;
                }
            }
            // Live pages, ascending vid = ascending dense slot; coalesce runs.
            const int32_t hw = pageTable_.size();
            const int32_t pb = global_->pageBytes();
            char* runStart = nullptr;
            int64_t runLen = 0;
            for (int32_t v = 0; v < hw; ++v) {
                char* page = pageTable_.peek(v);
                if (page == nullptr) continue;   // interior dead-vid hole
                if (runStart != nullptr && page == runStart + runLen) {
                    runLen += pb;
                } else {
                    if (runStart != nullptr) fill(runStart, runLen);
                    runStart = page;
                    runLen = pb;
                }
            }
            if (runStart != nullptr) fill(runStart, runLen);
        }

        // ---- Byte-bump scratch fill (the per-worker scratch / sealed path) --
        // A per-LB arena does not call these; the scratch arenas and the
        // sealed-page handoff fill bytes via allocBytes (a resolved `alloc`) +
        // mark / rewind / reset over the byte cursor, and are released per task,
        // never deloaded or compacted. They never touch the page tier, so they
        // never collide with `allocPage` containers on the same arena.

        /// @brief Allocate `bytes` contiguously and return a writable pointer —
        ///        the scratch fill-then-wrap allocation (a resolved `alloc`).
        ///
        /// @details
        /// A byte-bump `alloc` resolved to a `char*`: exact-length, never
        /// straddling a block (`alloc` pads to the next block boundary rather
        /// than cross it), so the bytes are one contiguous run. The pointer is
        /// valid until the next `rewind` past it, `reset`, or `releaseAll`: a
        /// scratch arena never deloads and never compacts while a task holds it,
        /// so nothing relocates the bytes under a cached pointer (the guarantee
        /// the scratch string views rely on).
        ///
        /// @param bytes Exact length; `0 < bytes <= blockBytes`.
        /// @return Pointer to `bytes` writable bytes.
        char* allocBytes(int32_t bytes);

        /// @brief Position token for stack-like reclamation: the byte-bump
        ///        cursor plus the generation it belongs to.
        ///
        /// @details
        /// Obtained from `mark()`, consumed by `rewind()`. The generation
        /// stamp makes a mark from before a wholesale `reset()` unusable —
        /// rewinding across a reset is a lifecycle bug and asserts.
        struct Mark {
            uint64_t generation;
            ArenaOffset cursor;       // byte-bump cursor at mark
        };

        /// @brief Capture the current bump position for a later `rewind`.
        ///
        /// @return Position token stamped with the current generation.
        Mark mark() const {
            return Mark{ generation_, cursor_ };
        }

        /// @brief Stack-like reclamation back to a captured position; the
        ///        freed span is poisoned.
        ///
        /// @details
        /// Legal only within the mark's generation and only backwards. Rolls
        /// the byte cursor back to the mark and poisons the reclaimed span
        /// (`popTo`); retained blocks stay retained — the next `allocBytes`
        /// reuses them. The across-reset generation assert guards the lifecycle.
        ///
        /// @param m A `mark()` taken in the current generation.
        void rewind(const Mark& m);

        /// @brief Wholesale per-scope reclamation: roll the byte cursor to 0,
        ///        poison everything used, bump the generation so every
        ///        outstanding view asserts. Blocks stay retained.
        void reset();

        /// @brief Generation counter — bumped by `reset` / `releaseAll`; the
        ///        escape-assert anchor for scratch string views.
        ///
        /// @return Current generation.
        uint64_t generation() const { return generation_; }

        /// @brief Lifetime high-water mark of `usedBytes()` — per-arena
        ///        sizing telemetry.
        ///
        /// @return Peak cursor position in bytes.
        int64_t peakUsedBytes() const { return peakUsedBytes_; }

        /// @brief Deep structural self-audit — asserts the arena's bookkeeping
        ///        (byte-bump and page tier) is self-consistent.
        ///
        /// @details
        /// The legitimate-assert centrepiece for the arena (Rule 19 / I-19): the
        /// three `PtrDirectory` tables (`blocks_` / `pageBlocks_` / `pageTable_`)
        /// are each internally consistent; the byte-bump block count exactly spans
        /// the cursor; `livePages_` equals the non-null `pageTable_` entries; the
        /// table tail is never a freed vid (`freePage` pops trailing nulls); the
        /// intrusive free-list length matches `freePageCount_`; and every carved
        /// page is accounted as either live or free. Under `GL_ARENA_PARANOID` it
        /// additionally proves every live and free page is distinct and lies in a
        /// carved block — no page is both free and live, no vid aliases another's
        /// page. Called UNCONDITIONALLY at the coarse, infrequent seams
        /// (`compactPages`, `releaseAll`, `markDeloaded` / `markResident`) and only
        /// under `GL_ARENA_PARANOID` at the fine seams (`allocPage` / `freePage`),
        /// so a violation aborts at the exact seam where the structure diverged.
        /// NDEBUG-gated (the body is empty under NDEBUG; this project keeps asserts
        /// on, so it is live in release).
        ///
        /// @invariant Pure read — never mutates the arena.
        void assertInvariants() const;

    private:
        /// @brief Compute the block geometry from the global manager on first
        ///        use.
        ///
        /// @details
        /// Deferred because LB objects can be constructed before the pool is
        /// initialized (the prover's `Memory` members are built before the
        /// config parse). Asserts the block size is a power of two — the
        /// shift / mask split requires it.
        void ensureGeometry();

        /// @brief Acquire blocks until `blockIndex` is backed by storage.
        ///
        /// @param blockIndex The block index the cursor has reached.
        void ensureBlock(int64_t blockIndex);

        /// @brief Carve or reuse one physical page for `allocPage`.
        ///
        /// @details
        /// Pops the free-list when non-empty; otherwise acquires a fresh block
        /// (`acquireBlock`) and carves it into `pageBytes` pages, pushing them
        /// onto the free-list, then pops one.
        ///
        /// @return A physical page free for binding to a fresh vid.
        char* takePage();

        /// @brief Bulk-acquire `count` blocks into a block directory — the raw
        ///        reload's one-mutex-per-chunk grant.
        ///
        /// @details
        /// Grants in fixed stack-buffer chunks (`GlobalMemoryManager::
        /// acquireBlocks`, one mutex acquisition per chunk) and pushes each into
        /// `dir` AFTER the lock is released (directory spill re-enters the
        /// manager, so it must not run under the pool mutex). For every real LB
        /// (< one chunk of blocks) this is a SINGLE mutex acquisition instead
        /// of one per block — the per-block loop measured ~2000 acquire/release
        /// round trips per big-LB deload/reload cycle. Heap-free (the chunk
        /// buffer is on the stack). Asserts residency.
        ///
        /// @param dir   The block directory (`blocks_` or `pageBlocks_`).
        /// @param count Blocks to acquire and append; >= 0 (0 = no-op).
        void acquireInto(PtrDirectory<kArenaBlockTableInline>& dir,
                         int32_t count);

        /// @brief Bulk-return every data block a directory holds — the bulk
        ///        twin of the per-block release loop, one mutex batch per
        ///        chunk instead of one per block.
        ///
        /// @details
        /// Reads the directory's block pointers in fixed stack-buffer chunks and
        /// returns each chunk with one `GlobalMemoryManager::releaseBlocks`
        /// call, so `releaseAll` returns a whole LB's blocks with a handful of
        /// mutex acquisitions instead of one per block. Does NOT clear `dir` —
        /// the caller does, which also returns the directory's own spilled
        /// storage. Release-identical to the former per-block loop.
        ///
        /// @param dir The block directory whose entries to return (`blocks_` or
        ///            `pageBlocks_`).
        void releaseFrom(PtrDirectory<kArenaBlockTableInline>& dir);

        GlobalMemoryManager* global_ = nullptr;  // null until bound
        int32_t blockBytes_ = 0;     // 0 until ensureGeometry; cached after
        int32_t blockShift_ = 0;     // log2(blockBytes_)
        uint32_t blockMask_ = 0;     // blockBytes_ - 1
        PtrDirectory<kArenaBlockTableInline> blocks_;  // byte-bump blocks, grant order
        ArenaOffset cursor_ = 0;     // forward bump position
        bool resident_ = true;
        uint64_t generation_ = 0;    // bumped by reset / releaseAll
        int64_t peakUsedBytes_ = 0;  // high-water of usedBytes()

        // Page tier (parallel to the byte bump above; see the page-tier section
        // of the public API). All three tables are pool-backed PtrDirectory drawn
        // from the same manager (small inline buffers keep tiny arenas
        // block-free); the free list is threaded intrusively through the free
        // pages themselves (no side container). The determinism contract
        // constrains element payloads and the per-LB virtual ids, not these
        // private tables (physical block identity stays invisible).
        PtrDirectory<kArenaBlockTableInline> pageBlocks_;  // blocks carved into pages
        PtrDirectory<kArenaPageTableInline> pageTable_;  // vid -> physical page (null=freed)
        char* freeHead_ = nullptr;       // free-page LIFO, threaded through pages
        int32_t freePageCount_ = 0;      // length of the freeHead_ chain
        int32_t livePages_ = 0;          // allocated minus freed
    };

}
