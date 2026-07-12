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

#include <atomic>
#include "extent_file.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace gl {

    /// @brief Extent-file growth chunk: when a new slab passes the file's end,
    ///        the file grows by whole multiples of this (default 256 MiB) so
    ///        growth is coarse, not per-slab.
    constexpr int64_t kExtentGrowChunkBytes = 256LL << 20;

    /// @brief Which of the three process-wide pools a `GlobalMemoryManager`
    ///        instance is — selects the exhaustion-assert knob name only.
    ///
    /// @details
    /// The four pools are otherwise identical code (one class, four
    /// instances); `kind` exists so a sizing problem points at the right
    /// `parameters.hpp` field. `Main` = the deloadable main pool
    /// (`static_pool_bytes`); `Persistent` = the never-deloaded second pool
    /// backing `Memory::intToBeProved` (`static_persistent_pool_bytes`);
    /// `Mail` = the never-deloaded third pool backing the cross-LB pull-model
    /// mail log (`static_mail_pool_bytes`); `Lb` = the never-deloaded fourth
    /// pool backing the LB object store (the `Memory` node shells themselves,
    /// `static_lb_pool_bytes`). A unit test's private instance is `Main` by
    /// default. Append-only: a future pool adds a new enumerator.
    enum class PoolKind { Main, Persistent, Mail, Lb };

    /// @brief Sizing triple of the statification memory hierarchy: one
    ///        program-start pool, carved into equal blocks, carved into equal
    ///        pages.
    ///
    /// @details
    /// The three values mirror the `prover_parameters` config fields
    /// `static_pool_bytes` / `static_block_bytes` / `static_page_bytes` and
    /// must satisfy `isValidStaticMemoryConfig` (pool a whole multiple of
    /// block, block a power of two) and `isValidStaticPageConfig` (block a
    /// whole multiple of page, page a power of two); all strictly positive.
    /// `GlobalMemoryManager::init` asserts that before reserving. `pageBytes`
    /// carries its default so the pool/block-only construction sites compile
    /// unchanged. `kind` selects which config knob the exhaustion assert
    /// names — `static_pool_bytes` (Main), `static_persistent_pool_bytes`
    /// (Persistent), or `static_mail_pool_bytes` (Mail) — and defaults
    /// `PoolKind::Main` so existing construction sites name the main knob
    /// unchanged.
    ///
    /// @invariant Immutable after the manager consumes it — the pool is
    ///            never grown or re-shaped (I-95).
    /// @see `isValidStaticMemoryConfig` / `isValidStaticPageConfig` in
    ///      `parameters.hpp`.
    struct StaticMemoryConfig {
        int64_t poolBytes;
        int32_t blockBytes;
        int32_t pageBytes = 8192;
        PoolKind kind = PoolKind::Main;
    };


    /// @brief Process-wide dispenser of fixed-size memory blocks carved from
    ///        ONE program-start reservation — the root of the statification
    ///        memory hierarchy.
    ///
    /// @details
    /// `init` makes a single reservation of `poolBytes` (the software model
    /// of the ASIC's fixed SRAM): never freed during the run, never grown.
    /// Per-LB memory managers call `acquireBlock` when they need storage and
    /// `releaseBlock` when an LB deloads to SSD or is destroyed; both run
    /// under an internal mutex because LBs are processed in parallel.
    ///
    /// Grant order under the mutex is nondeterministic. That is sound only
    /// under the statification determinism contract: NO observable behavior
    /// may depend on physical block identity or grant order — containers
    /// address storage through per-LB virtual offsets (resolved to a physical
    /// address only at the access site), and deloaded bytes are a pure
    /// function of logical content (I-107).
    ///
    /// Pool exhaustion is an assert naming the instance's configured pool
    /// knob (`static_pool_bytes` / `static_persistent_pool_bytes` /
    /// `static_mail_pool_bytes` — selected by `StaticMemoryConfig::kind`),
    /// never a fallback heap allocation — sizing problems must surface
    /// immediately, at their origin (I-19, I-95).
    ///
    /// Instantiable as a plain class so unit tests can build private
    /// instances with tiny pools; production code uses THREE process-wide
    /// instances — `staticMemory()` / `initStaticMemory()` (the deloadable
    /// main pool), `persistentMemory()` / `initPersistentMemory()` (the
    /// never-deloaded persistent pool that backs `Memory::intToBeProved`),
    /// and `mailMemory()` / `initMailMemory()` (the never-deloaded mail pool
    /// that backs the cross-LB pull-model mail log).
    ///
    /// @invariant Exactly one reservation per instance lifetime; `init` is
    ///            idempotent for an identical config and asserts on a
    ///            mismatching re-init.
    /// @invariant Every granted block is distinct, lies inside the pool, and
    ///            starts at a whole multiple of `blockBytes` from the pool
    ///            base.
    /// @see `LbArena` (the per-LB consumer), `StaticMemoryConfig`.
    class GlobalMemoryManager {
    public:
        GlobalMemoryManager() = default;

        /// @brief Frees the reservation (test instances only — the
        ///        process-wide instance lives until process exit).
        ///
        /// @details
        /// Deliberately does NOT assert `blocksInUse() == 0`: at process
        /// exit LBs are reclaimed by the OS rather than walked, so the
        /// process-wide instance legitimately dies with blocks outstanding.
        ~GlobalMemoryManager();

        GlobalMemoryManager(const GlobalMemoryManager&) = delete;
        GlobalMemoryManager& operator=(const GlobalMemoryManager&) = delete;
        GlobalMemoryManager(GlobalMemoryManager&&) = delete;
        GlobalMemoryManager& operator=(GlobalMemoryManager&&) = delete;

        /// @brief Make the one reservation and carve it into blocks.
        ///
        /// @details
        /// Asserts `isValidStaticMemoryConfig` on the triple. Idempotent
        /// when called again with a byte-identical config (run modes may
        /// construct more than one `ExpressionAnalyzer` per process);
        /// asserts on a mismatching re-init — two different sizings in one
        /// process is a configuration bug, not a request to re-shape.
        ///
        /// @param cfg Pool / block / page sizing triple (see
        ///            `StaticMemoryConfig`).
        /// @invariant After return, `initialized()` is true and the pool
        ///            base + shape never change for the instance lifetime.
        void init(const StaticMemoryConfig& cfg);

        /// @brief Whether `init` has run.
        ///
        /// @return `true` after the first successful `init`.
        bool initialized() const;

        /// @brief Grant one block (recycled first, else carved fresh).
        ///
        /// @details
        /// Thread-safe (mutex). Recycled blocks are re-granted before fresh
        /// ones so the carve cursor is a high-water mark of simultaneous
        /// demand. Exhaustion — every block carved and none recycled — is an
        /// assert naming `static_pool_bytes`; there is no fallback path.
        ///
        /// @return Pointer to the granted block of `blockBytes()` bytes.
        ///         Physical identity must not influence any observable
        ///         (I-107).
        char* acquireBlock();

        /// @brief Return a granted block to the recycle queue.
        ///
        /// @details
        /// Thread-safe (mutex). Asserts the pointer lies inside the pool,
        /// is block-aligned, and is currently granted (double release and
        /// foreign pointers are bugs we want at their origin).
        ///
        /// @param block A pointer previously returned by `acquireBlock`.
        void releaseBlock(char* block);

        /// @brief Grant `count` blocks under ONE mutex acquisition — the bulk
        ///        raw-image reload / eviction grant.
        ///
        /// @details
        /// Thread-safe: takes the pool mutex ONCE, grants `count` blocks
        /// (recycled first, else carved), and updates the accounting
        /// (`blocksInUse` / peak / grant ledger) and the one-shot grant trigger
        /// exactly as `count` separate `acquireBlock` calls would — the trigger
        /// fires once if the batch crosses the armed threshold, outside the
        /// mutex. It exists because the raw arena reload acquires a whole LB's
        /// blocks at once, and the per-block mutex traffic was a measured floor
        /// (~2000 acquire/release round trips per big-LB deload/reload cycle).
        /// Exhaustion of any block asserts naming the instance's pool knob, per
        /// `acquireBlock` (never a fallback, Rule 19). The output array is
        /// filled UNDER the mutex; the caller must NOT re-enter this manager
        /// from within a hypothetical callback — there is none, `out` is a plain
        /// array so directory spill (which does re-enter) happens after the
        /// lock is released.
        ///
        /// @param count Blocks to grant; >= 1.
        /// @param out   Caller array of at least `count` `char*`; filled with
        ///              the granted block pointers. Physical identity must not
        ///              influence any observable (I-107).
        void acquireBlocks(int32_t count, char** out);

        /// @brief Return `count` blocks under ONE mutex acquisition — the bulk
        ///        raw-image eviction / teardown release.
        ///
        /// @details
        /// Thread-safe: takes the pool mutex ONCE and returns every block in
        /// `blocks[0..count)` (each validated as `releaseBlock` would —
        /// in-pool, block-aligned, currently granted — asserts otherwise), then
        /// decrements `blocksInUse` by `count`. The bulk twin of `releaseBlock`
        /// for `LbArena::releaseAll`, so an LB's blocks return with one lock
        /// acquisition instead of one per block — the release half of the
        /// per-block mutex traffic the bulk grant exists to remove.
        ///
        /// @param blocks Array of `count` pointers previously returned by
        ///               `acquireBlock` / `acquireBlocks`.
        /// @param count  Blocks to return; >= 1.
        void releaseBlocks(char* const* blocks, int32_t count);

        /// @brief Bytes per block (the grant unit).
        ///
        /// @return `static_block_bytes` of the consumed config. Asserts
        ///         initialized; lock-free (immutable after `init`).
        int32_t blockBytes() const;

        /// @brief Bytes per page (the per-LB allocation unit).
        ///
        /// @details
        /// Owned here so every per-LB manager carves identically; the page
        /// size is a property of the hierarchy, not of one LB.
        ///
        /// @return `static_page_bytes` of the consumed config. Asserts
        ///         initialized; lock-free (immutable after `init`).
        int32_t pageBytes() const;

        /// @brief Total number of blocks the pool holds.
        ///
        /// @return `poolBytes / blockBytes`. Asserts initialized.
        int64_t totalBlocks() const;

        /// @brief Blocks currently granted (telemetry).
        ///
        /// @return Granted minus returned. Thread-safe.
        int64_t blocksInUse() const;

        /// @brief High-water mark of simultaneously granted blocks
        ///        (telemetry for sizing the config defaults).
        ///
        /// @return Peak of `blocksInUse()` over the instance lifetime.
        ///         Thread-safe.
        int64_t peakBlocksInUse() const;

        /// @brief Record a paged container's single -> two-level page-directory
        ///        promotion: bump the promotion count and the peak data-page
        ///        high-water.
        ///
        /// @details
        /// Process documentation only (Rule 16 / I-44): nothing in the prover
        /// reads it. `PagedVector` calls it once when its directory crosses
        /// `dirCap`; `PagedHashIndex` calls it once in `reset` when it builds a
        /// two-level directory. Lock-free (atomics, relaxed) — invoked from the
        /// single-threaded container write side, never the parallel burst. The
        /// count may shift run-to-run with the timing-dependent deload/reload
        /// set, so it is observability, not a determinism-gated value.
        ///
        /// @param numPages Data pages the container holds at the promotion.
        void recordTwoLevelPromotion(int32_t numPages);

        /// @brief Number of single -> two-level page-directory promotions across
        ///        all paged containers this instance has seen.
        ///
        /// @return The promotion count; `> 0` confirms some container's index
        ///         spilled past a single directory page. Lock-free.
        int64_t twoLevelPromotions() const;

        /// @brief Peak data-page count observed at any two-level promotion — the
        ///        deepest paged container this instance saw go two-level.
        ///
        /// @return The high-water data-page count; 0 if none promoted. Lock-free.
        int32_t peakPagesHeld() const;

        /// @brief Print the two-level page-directory telemetry to stderr — the
        ///        end-of-batch spill summary.
        ///
        /// @details
        /// One line: the promotion count + the peak data-page high-water. A
        /// forced-small-page run reports a non-zero count (real containers
        /// spilled into the two-level directory); the production 8 KiB page
        /// reports ~zero. Observability only.
        void reportPageStats() const;

        /// @brief Grants counted since the last `resetGrantLedger` — the
        ///        steward's only legal mid-iteration pressure signal.
        ///
        /// @details
        /// Within one kernel iteration block traffic is grants-only
        /// (block returns happen at barriers or through the steward's
        /// own planned work), so this counter is monotone and its value
        /// after k grants is independent of grant interleaving — unlike
        /// `blocksInUse()`, which dips as planned frees complete and is
        /// therefore timing-dependent mid-iteration
        /// (I-106). Thread-safe.
        ///
        /// @return Grants since the last ledger reset.
        int64_t grantsSinceBarrier() const;

        /// @brief Zero the grant ledger — called at the kernel's
        ///        end-of-iteration barrier, single-threaded.
        void resetGrantLedger();

        /// @brief Arm the once-per-iteration grant trigger: `fire` runs
        ///        when the ledger reaches `threshold`, then self-disarms.
        ///
        /// @details
        /// The crossing grant invokes `fire` AFTER releasing the
        /// manager's mutex, exactly once (the trigger disarms before the
        /// call). Determinism: the iteration's grant SET is
        /// deterministic, so whether the ledger reaches `threshold` in a
        /// given iteration is run-invariant — only WHICH thread's grant
        /// crosses varies, which is unobservable. `fire` must be
        /// lock-light (the steward's: set an atomic flag + notify) and
        /// must not call back into this manager. Asserts not already
        /// armed (re-arming without a barrier reset is a lifecycle bug)
        /// and a non-empty callable.
        ///
        /// @param threshold Ledger value that fires the trigger; >= 1
        ///                  (a zero-or-negative demand is decided at the
        ///                  barrier itself, never delegated here).
        /// @param fire      Callback run once by the crossing grant.
        void armGrantTrigger(int64_t threshold,
                             std::function<void()> fire);

        /// @brief Disarm the grant trigger — barrier cleanup; defined
        ///        no-op when the trigger already fired or was never
        ///        armed this iteration (both are completed lifecycles,
        ///        not failures).
        void disarmGrantTrigger();

        /// @brief Install (or clear, with an empty function) the FORENSIC
        ///        EXHAUSTION REPORTER — a diagnostic callback `grantLocked`
        ///        invokes at the wall, immediately BEFORE the exhaustion
        ///        assert fires.
        ///
        /// @details
        /// Pure observability at the point of death (the stuck-tripwire
        /// precedent: print, then assert — the assert REMAINS the failure,
        /// Rule 19 untouched; no fallback path is created). The prover
        /// installs it for the `prove()` scope (capturing the LB grid and
        /// the steward) and clears it on scope exit; a pool with no
        /// reporter installed asserts exactly as before. A `std::function`
        /// is deliberate: installation is a cold, single-threaded,
        /// once-per-prove operation, never on a grant path.
        ///
        /// CONTRACT — THE REPORTER RUNS WHILE THE POOL MUTEX IS HELD. It
        /// must never call back into this manager: no `acquireBlock` /
        /// `releaseBlock` / `acquireBlocks` / `releaseBlocks`, and none of
        /// the mutex-taking getters (`blocksInUse()`,
        /// `grantsSinceBarrier()`, `assignDeloadOrdinal()`, ...) — any of
        /// those deadlocks on the held mutex. The pool counters it needs
        /// are therefore PASSED IN (`blocksInUse`, `totalBlocks`, read
        /// under the already-held mutex at the call site). Legal reads
        /// inside the reporter: per-LB claim words (atomics), per-arena
        /// bookkeeping (`LbArena::blocksHeld()` / `resident()` — plain
        /// member reads that never touch this manager), the steward's
        /// relaxed in-flight atomics, `DeloadStats` atomics, and the
        /// scratch registries' per-slot arena bookkeeping. Reads of other
        /// threads' live arenas are RACY (the process is about to abort;
        /// the census is approximate by design).
        ///
        /// @param reporter Callback receiving (blocksInUse, totalBlocks) at
        ///                 the wall; empty clears the hook (a completed
        ///                 lifecycle, not a failure).
        void setExhaustionReporter(
            std::function<void(int64_t, int64_t)> reporter);

        /// @brief Invoke the installed exhaustion reporter under the pool
        ///        mutex — the unit-test seam for the death-path call
        ///        context.
        ///
        /// @details
        /// The real invocation site is `grantLocked` at the wall, followed
        /// by an assert that aborts the process — untestable in the
        /// in-tree harness (no death-test support; `assert` aborts the
        /// whole suite). This seam reproduces the exact calling context
        /// (pool mutex held, counters read under it) WITHOUT the abort, so
        /// a test can prove the census callback performs no manager
        /// re-entry (a violation deadlocks the test loudly) and receives
        /// the correct counter values. Asserts a reporter is installed —
        /// invoking a missing diagnostic is a test bug (production guards
        /// the call on installation).
        void invokeExhaustionReporterForTest();

        /// @brief Assign the next process-monotonic deload ordinal — the
        ///        injective per-LB file-name handle that replaced the
        ///        chain hash.
        ///
        /// @details
        /// Thread-safe (mutex). Post-increments one counter, so two LBs
        /// can never receive the same ordinal in one process — file
        /// identity is provably unique, no hash collision possible. The
        /// counter is process-monotonic and deliberately survives
        /// `resetDeloadRegistry` (a persisting LB keeps its ordinal across
        /// batches; reusing a number could collide a stale image with a
        /// freshly assigned one). A fresh process starts at 0, so two runs
        /// of the same input assign the same ordinals — the kernel barrier
        /// hands LBs to assignment in a deterministic order.
        ///
        /// @return The freshly assigned ordinal; >= 0, unique this process.
        int64_t assignDeloadOrdinal();

        /// @brief Record an `ordinal -> full LB chain` mapping for
        ///        `registry.txt` — the human/audit resolver of the numeric
        ///        file names.
        ///
        /// @details
        /// Thread-safe (mutex) — the parallel synchronous-eviction sweep
        /// registers from several worker threads. Idempotent: re-recording
        /// the same ordinal with the same chain is the normal case (every
        /// dump re-registers); recording an ordinal already bound to a
        /// DIFFERENT chain is an identity bug and asserts (Rule 19). The
        /// ordinal must already be assigned (`< nextDeloadOrdinal_`).
        ///
        /// @param ordinal A value returned by `assignDeloadOrdinal`.
        /// @param chain   The LB's full parent-chain string (its identity).
        void registerDeloadOrdinal(int64_t ordinal,
                                   const std::string& chain);

        /// @brief The accumulated `ordinal -> chain` map — the input to
        ///        `lbdeload::rewriteRegistry`.
        ///
        /// @details
        /// Returns a reference, NOT a copy: the sole caller is the kernel
        /// barrier fold, single-threaded after the steward quiesce and the
        /// synchronous-eviction join, so no concurrent mutation can race
        /// the read. `std::map` keeps the dump ascending by ordinal
        /// (deterministic output).
        ///
        /// @return The registry map (ascending ordinal).
        const std::map<int64_t, std::string>& deloadRegistry() const;

        /// @brief Clear the registry map at batch start so `registry.txt`
        ///        describes only the current `.deload/` contents.
        ///
        /// @details
        /// Clears the `ordinal -> chain` map (called next to
        /// `purgeDeloadDirectory`, which empties `.deload/`). Deliberately
        /// does NOT reset the ordinal counter: the counter stays
        /// process-monotonic so a persisting LB never collides a reused
        /// number (see `assignDeloadOrdinal`). Every LB that dumps in the
        /// new batch re-registers, so the cleared map refills completely.
        void resetDeloadRegistry();

        /// @brief Open (creating) the ONE extent file for the v4 raw-eviction
        ///        images and arm the extent datapath for this batch.
        ///
        /// @details
        /// Closes any prior batch's extent file, opens the new one,
        /// preallocates it to `initialBytes`, (re)initializes the slab
        /// allocator to the pool block granularity, and BUMPS the extent
        /// EPOCH so any LB carrying a slab from a purged prior batch is treated
        /// as slab-less (its stale `rawExtentOffset_` is ignored — offsets
        /// recycled from 0 at reset). Sets `useExtent()` true. Called at batch
        /// start (after `purgeDeloadDirectory` deletes the old file). Only when
        /// `parameters.enable_extent_deload`; otherwise the named-file raw path
        /// stays active.
        ///
        /// @param path         The extent file path (under the deload dir).
        /// @param initialBytes Preallocated size (default 1.5x the pool).
        void openExtentFile(const std::filesystem::path& path,
                            int64_t initialBytes);

        /// @brief Close the extent file (if open) and disarm the datapath.
        ///
        /// @details
        /// Called BEFORE `purgeDeloadDirectory` (which deletes the file — an
        /// open handle would block the delete on Windows) and at process
        /// teardown via the destructor. Sets `useExtent()` false. A no-op when
        /// already closed.
        void closeExtentFile();

        /// @brief Whether the extent datapath is armed this batch.
        /// @return True between `openExtentFile` and `closeExtentFile`.
        bool useExtent() const {
            return extentEnabled_.load(std::memory_order_relaxed);
        }

        /// @brief The current extent epoch — bumped at every `openExtentFile`.
        ///
        /// @details
        /// An LB's cached slab (`rawExtentOffset_`) is valid ONLY if the LB's
        /// `rawExtentEpoch_` equals this; a mismatch means the slab is from a
        /// purged batch (offsets since recycled from 0) and must be ignored.
        /// Read on the dump path; only mutated single-threaded at batch start.
        ///
        /// @return The epoch id.
        int64_t extentEpoch() const { return extentEpoch_; }

        /// @brief The open extent file (positioned I/O at disjoint offsets).
        /// @return A reference for the raw dump/load positioned calls.
        PositionedFile& extentFile() { return extentFile_; }

        /// @brief Allocate an extent slab for an image of `need` bytes, growing
        ///        the file if the high-water passed its end.
        ///
        /// @details
        /// Serializes the slab allocation AND the file grow under one mutex so
        /// the `PositionedFile::grow` targets stay monotone under concurrent
        /// evictions (two threads racing the high-water must not order two
        /// grows backwards). The positioned WRITE that follows runs OUTSIDE
        /// this lock (disjoint offsets). Records the slab under its ordinal in
        /// the slab registry (the `registry.txt` extent audit columns).
        ///
        /// @param ordinal The LB's deload ordinal (slab-registry key).
        /// @param need    The image byte count to hold.
        /// @return The slab offset and its class capacity.
        SlabAllocation allocateExtentSlab(int64_t ordinal, int64_t need);

        /// @brief Return an extent slab to its class free-list and drop its
        ///        slab-registry row (class promotion / LB discharge).
        ///
        /// @param ordinal    The owning LB's deload ordinal (registry key).
        /// @param offset     The slab offset from `allocateExtentSlab`.
        /// @param classBytes The slab's class capacity.
        void freeExtentSlab(int64_t ordinal, int64_t offset,
                            int64_t classBytes);

        /// @brief The per-ordinal slab placements — the extent audit columns of
        ///        `lbdeload::rewriteRegistry`.
        ///
        /// @details
        /// Returns a reference, NOT a copy, on the same justification as
        /// `deloadRegistry()`: the sole production caller is the kernel barrier
        /// fold, single-threaded after the steward quiesce, so no executor
        /// mutation races the read. Reload never consults it (the LB's own
        /// `rawExtentOffset_` is the source of truth); it exists for the
        /// human-audit registry file only.
        ///
        /// @return Ordinal → current slab (ascending ordinal).
        const std::map<int64_t, SlabAllocation>& extentSlabRegistry() const {
            return extentSlabRegistry_;
        }

        /// @brief Σ live slab class sizes (the extent internal-slack numerator).
        /// @return `extentAllocatedBytes` telemetry.
        int64_t extentAllocatedBytes() const {
            return extentAllocator_.allocatedBytes();
        }

        /// @brief The extent file's current logical size.
        /// @return `extentFileBytes` telemetry.
        int64_t extentFileBytes() const { return extentFile_.fileSize(); }

        /// @brief Adjust the extent live-occupancy total by a signed delta.
        ///
        /// @details
        /// The dump wiring calls this with `newImage - lastImage` on each
        /// extent dump (the LB's occupancy replaces its prior) and `-lastImage`
        /// on slab free. Batch-scoped: zeroed by `openExtentFile`, so it stays
        /// correct across the several prove() calls of one batch (unlike the
        /// per-prove `DeloadStats` reset). One relaxed accumulation.
        ///
        /// @param delta Signed byte delta.
        void addExtentLive(int64_t delta) {
            extentLiveBytes_.fetch_add(delta, std::memory_order_relaxed);
        }

        /// @brief Σ image bytes of currently-Dumped raw LBs (the slack numerator).
        /// @return `extentLiveBytes` telemetry.
        int64_t extentLiveBytes() const {
            return extentLiveBytes_.load(std::memory_order_relaxed);
        }

    private:
        /// @brief Dequeue-or-carve one block and flag it granted; the
        ///        accounting-free core shared by both grant paths. Caller
        ///        holds `mutex_`.
        ///
        /// @return The granted block.
        char* grantLocked();

        /// @brief Validate and un-flag a returned block, park it on the
        ///        recycle queue; the accounting-free core shared by both
        ///        release paths. Caller holds `mutex_`.
        ///
        /// @param block The block being returned.
        void returnLocked(char* block);

        mutable std::mutex mutex_;
        char* pool_ = nullptr;
        StaticMemoryConfig cfg_{ 0, 0, 0 };
        int64_t totalBlocks_ = 0;
        int64_t carveCursor_ = 0;          // next never-granted block index
        std::deque<char*> recycled_;       // returned blocks awaiting re-grant
        std::vector<bool> granted_;        // per-block grant flag (assert aid)
        int64_t blocksInUse_ = 0;
        int64_t peakBlocksInUse_ = 0;
        // Two-level page-directory telemetry (process documentation, Rule 16 /
        // I-44; never a prover input). Atomic so the single-threaded container
        // write side records without taking the pool mutex.
        std::atomic<int64_t> twoLevelPromotions_{ 0 };
        std::atomic<int32_t> peakPagesHeld_{ 0 };
        int64_t grantsSinceBarrier_ = 0;   // monotone within an iteration
        int64_t grantTriggerThreshold_ = -1;  // -1 = disarmed
        std::function<void()> grantTriggerFire_;
        // Forensic exhaustion reporter (diagnostic only): invoked by
        // grantLocked at the wall, under mutex_, BEFORE the exhaustion
        // assert; empty = not installed. See setExhaustionReporter.
        std::function<void(int64_t, int64_t)> exhaustionReporter_;
        int64_t nextDeloadOrdinal_ = 0;    // process-monotonic deload file id
        std::map<int64_t, std::string> deloadRegistry_;  // ordinal -> chain

        // v4 raw-eviction extent file (D-195 extent
        // datapath). ONE preallocated file per batch; each LB owns a slab.
        // extentMutex_ serializes slab alloc + file grow (grow must stay
        // monotone); the positioned writes/reads run outside it (disjoint
        // offsets). extentEnabled_ is the A/B gate (relaxed-read on the dump
        // path); extentEpoch_ invalidates cross-batch stale slabs.
        PositionedFile extentFile_;
        ExtentAllocator extentAllocator_;
        std::mutex extentMutex_;
        std::atomic<bool> extentEnabled_{ false };
        int64_t extentEpoch_ = 0;
        std::atomic<int64_t> extentLiveBytes_{ 0 };
        // ordinal -> current slab, for registry.txt's audit columns only
        // (reload uses the LB's own rawExtentOffset_). Under extentMutex_.
        std::map<int64_t, SlabAllocation> extentSlabRegistry_;
    };

    /// @brief The process-wide static-memory manager instance.
    ///
    /// @details
    /// Production binding for `Memory` / per-LB managers. Unit tests build
    /// private `GlobalMemoryManager` instances instead and leave this one
    /// to the harness-level init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initStaticMemory` before any block traffic).
    GlobalMemoryManager& staticMemory();

    /// @brief Initialize the process-wide manager from a config triple.
    ///
    /// @details
    /// Thin forwarder to `staticMemory().init(cfg)`; carries the same
    /// idempotent-same-config / assert-on-mismatch contract. Called by the
    /// `ExpressionAnalyzer` constructor right after the JSON parameter
    /// parse, and by the unit-test harness with a small test pool.
    ///
    /// @param cfg Pool / block / page sizing triple.
    void initStaticMemory(const StaticMemoryConfig& cfg);

    /// @brief The process-wide PERSISTENT static-memory manager instance — the
    ///        second pool that backs `Memory::intToBeProved`.
    ///
    /// @details
    /// A separate reservation from `staticMemory()`, with smaller blocks
    /// (`static_persistent_block_bytes`). Each LB's persistent `LbArena` draws
    /// from this pool and is NEVER deloaded — its content stays resident for the
    /// LB's whole active life and is reclaimed only at discharge — so the
    /// deactivation survey can read `intToBeProved` regardless of the main
    /// arena's deload state (the determinism fix). Distinct pool, distinct
    /// telemetry, distinct exhaustion assert (`static_persistent_pool_bytes`).
    /// Unit tests build private instances and leave this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initPersistentMemory` before any persistent block traffic).
    /// @see `staticMemory` (the deloadable main pool), `initPersistentMemory`.
    GlobalMemoryManager& persistentMemory();

    /// @brief Initialize the process-wide persistent manager from a config
    ///        triple (must carry `isPersistent == true`).
    ///
    /// @details
    /// Thin forwarder to `persistentMemory().init(cfg)`; same idempotent-same-
    /// config / assert-on-mismatch contract as `initStaticMemory`. Called by the
    /// `ExpressionAnalyzer` constructor right after `initStaticMemory`, and by
    /// the unit-test harness with a small test pool.
    ///
    /// @param cfg Persistent pool / block / page sizing triple; its `kind`
    ///            must be `PoolKind::Persistent` so the exhaustion assert
    ///            names the right knob.
    void initPersistentMemory(const StaticMemoryConfig& cfg);

    /// @brief The process-wide MAIL static-memory manager instance — the
    ///        third pool that backs the cross-LB pull-model mail log.
    ///
    /// @details
    /// A separate reservation from both `staticMemory()` and
    /// `persistentMemory()`, NEVER deloaded: it holds every LB's committed
    /// mail batches and the per-(recipient, ancestor) ingestion cursors for a
    /// whole execution batch. A stand-alone pool by design — nothing reads its
    /// grant ledger, so it plays no role in any deload / throttle / steward
    /// decision, and the mail content never competes with the deloadable main
    /// pool. The single `ExpressionAnalyzer`-owned `mailArena` draws blocks
    /// from it; the `MailLog` containers pack pages out of that arena. Distinct
    /// pool, distinct telemetry, distinct exhaustion assert
    /// (`static_mail_pool_bytes`). Unit tests build private instances and leave
    /// this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initMailMemory` before any mail block traffic).
    /// @see `staticMemory` (the deloadable main pool), `persistentMemory`
    ///      (the second pool), `initMailMemory`.
    GlobalMemoryManager& mailMemory();

    /// @brief Initialize the process-wide mail manager from a config triple
    ///        (must carry `kind == PoolKind::Mail`).
    ///
    /// @details
    /// Thin forwarder to `mailMemory().init(cfg)`; same idempotent-same-config
    /// / assert-on-mismatch contract as `initStaticMemory`. Asserts the config
    /// carries `PoolKind::Mail` so the exhaustion assert names the right knob.
    /// Called by the `ExpressionAnalyzer` constructor right after
    /// `initPersistentMemory`, and by the unit-test harness with a small test
    /// pool.
    ///
    /// @param cfg Mail pool / block / page sizing triple; `kind` must be
    ///            `PoolKind::Mail`.
    void initMailMemory(const StaticMemoryConfig& cfg);

    /// @brief The process-wide LB-BODY static-memory manager instance — the
    ///        fourth pool that backs the LB object store.
    ///
    /// @details
    /// A separate reservation from the three pools above, NEVER deloaded: it
    /// backs the `LbStore` on the `ExpressionAnalyzer`, which places every
    /// `Memory` node object into a fixed-size slot carved from this pool's
    /// blocks instead of the malloc heap. A stand-alone pool by design —
    /// nothing reads its grant ledger, so it plays no role in any deload /
    /// throttle / steward decision, and the shells never compete with the
    /// deloadable main pool. The `LbStore` carves each block into
    /// `sizeof(Memory)`-sized slots (it draws blocks, not pages). Distinct pool,
    /// distinct telemetry, distinct exhaustion assert (`static_lb_pool_bytes`).
    /// Unit tests build private instances and leave this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initLbMemory` before any LB-store block traffic).
    /// @see `staticMemory` (the deloadable main pool), `persistentMemory`
    ///      (the second pool), `mailMemory` (the third pool), `initLbMemory`.
    GlobalMemoryManager& lbMemory();

    /// @brief Initialize the process-wide LB-body manager from a config triple
    ///        (must carry `kind == PoolKind::Lb`).
    ///
    /// @details
    /// Thin forwarder to `lbMemory().init(cfg)`; same idempotent-same-config
    /// / assert-on-mismatch contract as `initStaticMemory`. Asserts the config
    /// carries `PoolKind::Lb` so the exhaustion assert names the right knob.
    /// Called by the `ExpressionAnalyzer` constructor right after
    /// `initMailMemory`, and by the unit-test harness with a small test pool.
    ///
    /// @param cfg LB-body pool / block / page sizing triple; `kind` must be
    ///            `PoolKind::Lb`.
    void initLbMemory(const StaticMemoryConfig& cfg);

}
