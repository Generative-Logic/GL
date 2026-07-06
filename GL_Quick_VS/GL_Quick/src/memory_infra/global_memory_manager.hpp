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
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace gl {

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
        int64_t nextDeloadOrdinal_ = 0;    // process-monotonic deload file id
        std::map<int64_t, std::string> deloadRegistry_;  // ordinal -> chain
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
