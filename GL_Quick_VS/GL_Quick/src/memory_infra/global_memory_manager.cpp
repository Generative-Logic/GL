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

#include "global_memory_manager.hpp"
#include "../infra/diagnostics_log.hpp"

#include "../parameters.hpp"

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <new>
#include <thread>

namespace gl {

    /// @brief Frees the reservation (test instances only — the process-wide
    ///        instance lives until process exit).
    ///
    /// @details
    /// Deliberately does NOT assert `blocksInUse() == 0`: at process exit
    /// LBs are reclaimed by the OS rather than walked, so the process-wide
    /// instance legitimately dies with blocks outstanding.
    GlobalMemoryManager::~GlobalMemoryManager() {
        ::operator delete(pool_);
    }

    /// @brief Make the one reservation and carve it into blocks.
    ///
    /// @details
    /// Asserts `isValidStaticMemoryConfig` and `isValidStaticPageConfig` on
    /// the sizing triple. Idempotent when
    /// called again with a byte-identical config (run modes may construct
    /// more than one `ExpressionAnalyzer` per process); asserts on a
    /// mismatching re-init — two different sizings in one process is a
    /// configuration bug, not a request to re-shape.
    ///
    /// @param cfg Pool / block / page sizing triple (see
    ///            `StaticMemoryConfig`).
    /// @invariant After return, `initialized()` is true and the pool base +
    ///            shape never change for the instance lifetime.
    void GlobalMemoryManager::init(const StaticMemoryConfig& cfg) {
        assert(isValidStaticMemoryConfig(cfg.poolBytes, cfg.blockBytes)
            && isValidStaticPageConfig(cfg.blockBytes, cfg.pageBytes));
        std::lock_guard<std::mutex> lock(mutex_);
        if (pool_ != nullptr) {
            assert(cfg.poolBytes == cfg_.poolBytes
                && cfg.blockBytes == cfg_.blockBytes
                && cfg.pageBytes == cfg_.pageBytes
                && "GlobalMemoryManager re-init with a different sizing "
                   "triple — one process, one static-memory shape");
            return;
        }
        cfg_ = cfg;
        totalBlocks_ = cfg.poolBytes / cfg.blockBytes;
        pool_ = static_cast<char*>(
            ::operator new(static_cast<std::size_t>(cfg.poolBytes)));
        granted_.assign(static_cast<std::size_t>(totalBlocks_), false);
    }

    /// @brief Whether `init` has run.
    ///
    /// @return `true` after the first successful `init`.
    bool GlobalMemoryManager::initialized() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return pool_ != nullptr;
    }

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
    char* GlobalMemoryManager::acquireBlock() {
        char* block = nullptr;
        std::function<void()> fire;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(pool_ != nullptr);
            block = grantLocked();
            ++blocksInUse_;
            if (blocksInUse_ > peakBlocksInUse_)
                peakBlocksInUse_ = blocksInUse_;
            ++grantsSinceBarrier_;
            if (grantTriggerThreshold_ >= 0
                && grantsSinceBarrier_ >= grantTriggerThreshold_) {
                // Self-disarm BEFORE the call: exactly-once, and the
                // callback runs outside the mutex (it notifies the
                // steward and must never re-enter this manager).
                fire = std::move(grantTriggerFire_);
                grantTriggerFire_ = nullptr;
                grantTriggerThreshold_ = -1;
            }
        }
        if (fire) fire();
        return block;
    }

    /// @brief Grants counted since the last `resetGrantLedger` — the
    ///        steward's only legal mid-iteration pressure signal.
    ///
    /// @details
    /// Within one kernel iteration block traffic is grants-only, so this
    /// counter is monotone and order-independent — unlike `blocksInUse()`,
    /// which dips as planned frees complete (I-106).
    /// Thread-safe.
    ///
    /// @return Grants since the last ledger reset.
    int64_t GlobalMemoryManager::grantsSinceBarrier() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return grantsSinceBarrier_;
    }

    /// @brief Zero the grant ledger — called at the kernel's
    ///        end-of-iteration barrier, single-threaded.
    void GlobalMemoryManager::resetGrantLedger() {
        std::lock_guard<std::mutex> lock(mutex_);
        grantsSinceBarrier_ = 0;
    }

    /// @brief Arm the once-per-iteration grant trigger: `fire` runs when
    ///        the ledger reaches `threshold`, then self-disarms.
    ///
    /// @details
    /// The crossing grant invokes `fire` after releasing the manager's
    /// mutex, exactly once. Asserts not already armed and a non-empty
    /// callable; `threshold >= 1` (a zero-or-negative demand is decided
    /// at the barrier itself, never delegated to the trigger).
    ///
    /// @param threshold Ledger value that fires the trigger; >= 1.
    /// @param fire      Callback run once by the crossing grant.
    void GlobalMemoryManager::armGrantTrigger(int64_t threshold,
                                              std::function<void()> fire) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(threshold >= 1
            && "armGrantTrigger with a non-positive threshold — the "
               "barrier decides immediate demand itself");
        assert(grantTriggerThreshold_ < 0
            && "armGrantTrigger while armed — re-arm requires a barrier "
               "reset first");
        assert(fire && "armGrantTrigger with an empty callback");
        grantTriggerThreshold_ = threshold;
        grantTriggerFire_ = std::move(fire);
    }

    /// @brief Disarm the grant trigger — barrier cleanup; defined no-op
    ///        when the trigger already fired or was never armed this
    ///        iteration (both are completed lifecycles, not failures).
    void GlobalMemoryManager::disarmGrantTrigger() {
        std::lock_guard<std::mutex> lock(mutex_);
        grantTriggerThreshold_ = -1;
        grantTriggerFire_ = nullptr;
    }

    /// @brief Assign the next process-monotonic deload ordinal — the
    ///        injective per-LB file-name handle that replaced the chain
    ///        hash.
    ///
    /// @details
    /// Thread-safe (mutex). Post-increments one counter, so two LBs can
    /// never receive the same ordinal in one process — file identity is
    /// provably unique, no hash collision possible. Process-monotonic and
    /// deliberately not reset per batch (a persisting LB keeps its ordinal;
    /// reuse could collide a stale image with a fresh one). A fresh process
    /// starts at 0, so two runs of the same input assign the same ordinals.
    ///
    /// @return The freshly assigned ordinal; >= 0, unique this process.
    int64_t GlobalMemoryManager::assignDeloadOrdinal() {
        std::lock_guard<std::mutex> lock(mutex_);
        return nextDeloadOrdinal_++;
    }

    /// @brief Install (or clear) the forensic exhaustion reporter (see the
    ///        header for the full under-mutex contract).
    ///
    /// @param reporter Callback receiving (blocksInUse, totalBlocks) at the
    ///                 wall; empty clears the hook.
    void GlobalMemoryManager::setExhaustionReporter(
        std::function<void(int64_t, int64_t)> reporter) {
        std::lock_guard<std::mutex> lock(mutex_);
        exhaustionReporter_ = std::move(reporter);
    }

    /// @brief Invoke the installed exhaustion reporter under the pool mutex
    ///        — the unit-test seam for the death-path call context (see the
    ///        header for why the death path itself is untestable).
    void GlobalMemoryManager::invokeExhaustionReporterForTest() {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(exhaustionReporter_
            && "invokeExhaustionReporterForTest without an installed "
               "reporter");
        exhaustionReporter_(blocksInUse_, totalBlocks_);
    }

    /// @brief Record an `ordinal -> full LB chain` mapping for
    ///        `registry.txt` — the human/audit resolver of the numeric file
    ///        names.
    ///
    /// @details
    /// Thread-safe (mutex) — the parallel synchronous-eviction sweep
    /// registers from several worker threads. Idempotent: re-recording the
    /// same ordinal with the same chain is the normal case (every dump
    /// re-registers); an ordinal already bound to a DIFFERENT chain is an
    /// identity bug and asserts (Rule 19). The ordinal must already be
    /// assigned.
    ///
    /// @param ordinal A value returned by `assignDeloadOrdinal`.
    /// @param chain   The LB's full parent-chain string (its identity).
    void GlobalMemoryManager::registerDeloadOrdinal(
        int64_t ordinal, const std::string& chain) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(ordinal >= 0 && ordinal < nextDeloadOrdinal_
            && "registerDeloadOrdinal with an unassigned ordinal");
        const auto it = deloadRegistry_.find(ordinal);
        assert((it == deloadRegistry_.end() || it->second == chain)
            && "deload ordinal bound to two different LB chains");
        deloadRegistry_[ordinal] = chain;
    }

    /// @brief The accumulated `ordinal -> chain` map — the input to
    ///        `lbdeload::rewriteRegistry`.
    ///
    /// @details
    /// Returns a reference, NOT a copy: the sole caller is the kernel
    /// barrier fold, single-threaded after the steward quiesce and the
    /// synchronous-eviction join, so no concurrent mutation races the read.
    /// `std::map` keeps the dump ascending by ordinal.
    ///
    /// @return The registry map (ascending ordinal).
    const std::map<int64_t, std::string>&
    GlobalMemoryManager::deloadRegistry() const {
        return deloadRegistry_;
    }

    /// @brief Clear the registry map at batch start so `registry.txt`
    ///        describes only the current `.deload/` contents.
    ///
    /// @details
    /// Clears the `ordinal -> chain` map (called next to
    /// `purgeDeloadDirectory`). Deliberately does NOT reset the ordinal
    /// counter: it stays process-monotonic so a persisting LB never
    /// collides a reused number. Every LB that dumps in the new batch
    /// re-registers, so the cleared map refills completely.
    void GlobalMemoryManager::resetDeloadRegistry() {
        std::lock_guard<std::mutex> lock(mutex_);
        deloadRegistry_.clear();
    }

    /// @brief Open the extent file and arm the raw-eviction datapath (see the
    ///        declaration for the full contract).
    ///
    /// @param path         The extent file path.
    /// @param initialBytes Preallocated size.
    void GlobalMemoryManager::openExtentFile(const std::filesystem::path& path,
                                             int64_t initialBytes) {
        std::lock_guard<std::mutex> lock(extentMutex_);
        // A prior batch's file is closed here as a defined batch-reset step
        // (the purge already deleted its bytes). Not defensive — the batch
        // boundary resets extent state exactly as it resets the registry.
        if (extentFile_.isOpen()) extentFile_.close();
        extentFile_.open(path);
        extentFile_.preallocate(initialBytes);
        extentAllocator_.init(blockBytes());   // offsets recycle from 0
        extentSlabRegistry_.clear();
        extentLiveBytes_.store(0, std::memory_order_relaxed);
        ++extentEpoch_;                        // invalidate cross-batch slabs
        extentEnabled_.store(true, std::memory_order_relaxed);
    }

    /// @brief Close the extent file and disarm the datapath (see the
    ///        declaration).
    void GlobalMemoryManager::closeExtentFile() {
        std::lock_guard<std::mutex> lock(extentMutex_);
        extentEnabled_.store(false, std::memory_order_relaxed);
        if (extentFile_.isOpen()) extentFile_.close();
    }

    /// @brief Allocate an extent slab, grow the file if needed, and record the
    ///        slab under its ordinal (see the declaration).
    ///
    /// @param ordinal The LB's deload ordinal (slab-registry key).
    /// @param need    The image byte count to hold.
    /// @return The slab offset and class capacity.
    SlabAllocation GlobalMemoryManager::allocateExtentSlab(int64_t ordinal,
                                                           int64_t need) {
        assert(ordinal >= 0 && "extent slab for an unassigned ordinal");
        std::lock_guard<std::mutex> lock(extentMutex_);
        const SlabAllocation slab = extentAllocator_.allocSlab(need);
        const int64_t hw = extentAllocator_.highWaterBytes();
        if (hw > extentFile_.fileSize()) {
            const int64_t grown =
                (hw + kExtentGrowChunkBytes - 1)
                / kExtentGrowChunkBytes * kExtentGrowChunkBytes;
            extentFile_.grow(grown);
        }
        extentSlabRegistry_[ordinal] = slab;
        return slab;
    }

    /// @brief Return an extent slab to its class free-list and drop its
    ///        slab-registry row (see the declaration).
    ///
    /// @param ordinal    The owning LB's deload ordinal.
    /// @param offset     The slab offset.
    /// @param classBytes The slab class capacity.
    void GlobalMemoryManager::freeExtentSlab(int64_t ordinal, int64_t offset,
                                             int64_t classBytes) {
        std::lock_guard<std::mutex> lock(extentMutex_);
        const auto it = extentSlabRegistry_.find(ordinal);
        assert(it != extentSlabRegistry_.end()
            && it->second.offset == offset
            && it->second.classBytes == classBytes
            && "extent slab free does not match the registered slab");
        extentSlabRegistry_.erase(it);
        extentAllocator_.freeSlab(offset, classBytes);
    }

    /// @brief Return a granted block to the recycle queue.
    ///
    /// @details
    /// Thread-safe (mutex). Asserts the pointer lies inside the pool, is
    /// block-aligned, and is currently granted (double release and foreign
    /// pointers are bugs we want at their origin).
    ///
    /// @param block A pointer previously returned by `acquireBlock`.
    void GlobalMemoryManager::releaseBlock(char* block) {
        std::lock_guard<std::mutex> lock(mutex_);
        returnLocked(block);
        --blocksInUse_;
    }

    /// @brief Grant `count` blocks under ONE mutex acquisition — the bulk
    ///        raw-image reload / eviction grant.
    ///
    /// @details
    /// One lock, `count` `grantLocked` calls, one accounting update, one grant-
    /// trigger check — accounting-identical to `count` separate `acquireBlock`
    /// calls, but the mutex is taken once. The trigger fires once outside the
    /// mutex when the batch crosses the armed threshold.
    ///
    /// @param count Blocks to grant; >= 1.
    /// @param out   Caller array of at least `count` `char*`; filled.
    void GlobalMemoryManager::acquireBlocks(int32_t count, char** out) {
        assert(count >= 1);
        std::function<void()> fire;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(pool_ != nullptr);
            for (int32_t i = 0; i < count; ++i)
                out[i] = grantLocked();
            blocksInUse_ += count;
            if (blocksInUse_ > peakBlocksInUse_)
                peakBlocksInUse_ = blocksInUse_;
            grantsSinceBarrier_ += count;
            if (grantTriggerThreshold_ >= 0
                && grantsSinceBarrier_ >= grantTriggerThreshold_) {
                fire = std::move(grantTriggerFire_);
                grantTriggerFire_ = nullptr;
                grantTriggerThreshold_ = -1;
            }
        }
        if (fire) fire();
    }

    /// @brief Return `count` blocks under ONE mutex acquisition — the bulk
    ///        raw-image eviction / teardown release.
    ///
    /// @details
    /// One lock, `count` `returnLocked` calls (each validated in-pool /
    /// block-aligned / granted), one accounting decrement — release-identical
    /// to `count` separate `releaseBlock` calls with the mutex taken once.
    ///
    /// @param blocks Array of `count` previously granted pointers.
    /// @param count  Blocks to return; >= 1.
    void GlobalMemoryManager::releaseBlocks(char* const* blocks,
                                            int32_t count) {
        assert(count >= 1);
        std::lock_guard<std::mutex> lock(mutex_);
        for (int32_t i = 0; i < count; ++i)
            returnLocked(blocks[i]);
        blocksInUse_ -= count;
    }

    /// @brief Dequeue-or-carve one block and flag it granted; the
    ///        accounting-free core shared by both grant paths. Caller holds
    ///        `mutex_`.
    ///
    /// @details
    /// Recycled blocks are re-granted before fresh ones so the carve cursor
    /// is a high-water mark of simultaneous demand across BOTH grant paths.
    /// Exhaustion — every block carved and none recycled — is an assert
    /// naming the instance's pool sizing constant (`static_pool_bytes`,
    /// `static_persistent_pool_bytes`, or `static_mail_pool_bytes`, per
    /// `cfg_.kind`); there is no fallback path.
    ///
    /// @return The granted block.
    char* GlobalMemoryManager::grantLocked() {
        char* block = nullptr;
        if (!recycled_.empty()) {
            block = recycled_.front();
            recycled_.pop_front();
        }
        else {
            if (carveCursor_ >= totalBlocks_) {
                // THE WALL. Print the pool-side counters and run the
                // installed forensic reporter BEFORE the assert below fires
                // (the stuck-tripwire precedent: print, then assert — the
                // assert remains the failure, Rule 19 untouched). All reads
                // here are direct members under the already-held mutex; the
                // reporter receives the counters it must not fetch itself
                // (its contract forbids manager re-entry — see
                // setExhaustionReporter).
                std::cerr << "[EXHAUSTION] pool kind="
                          << static_cast<int>(cfg_.kind)
                          << " totalBlocks=" << totalBlocks_
                          << " blocksInUse=" << blocksInUse_
                          << " carveCursor=" << carveCursor_
                          << " recycled=" << recycled_.size()
                          << " peak=" << peakBlocksInUse_
                          << " blockBytes=" << cfg_.blockBytes << std::endl;
                if (exhaustionReporter_)
                    exhaustionReporter_(blocksInUse_, totalBlocks_);
            }
            // Exhaustion asserts naming the instance's pool sizing constant
            // (Rule 19 — never a fallback). Each pool names its own constants
            // so a sizing problem points at the right parameters.hpp field.
            switch (cfg_.kind) {
            case PoolKind::Persistent:
                assert(carveCursor_ < totalBlocks_
                    && "static-memory persistent pool exhausted — raise "
                       "static_persistent_pool_bytes (or lower "
                       "static_persistent_block_bytes) in parameters.hpp");
                break;
            case PoolKind::Mail:
                assert(carveCursor_ < totalBlocks_
                    && "static-memory mail pool exhausted — raise "
                       "static_mail_pool_bytes (or lower "
                       "static_mail_block_bytes) in parameters.hpp");
                break;
            case PoolKind::Lb:
                assert(carveCursor_ < totalBlocks_
                    && "static-memory LB-body pool exhausted — raise "
                       "static_lb_pool_bytes (or lower "
                       "static_lb_block_bytes) in parameters.hpp");
                break;
            case PoolKind::Main:
                assert(carveCursor_ < totalBlocks_
                    && "static-memory pool exhausted — raise static_pool_bytes "
                       "(or lower static_block_bytes) in parameters.hpp");
                break;
            }
            block = pool_ + carveCursor_ * cfg_.blockBytes;
            ++carveCursor_;
        }
        const int64_t idx = (block - pool_) / cfg_.blockBytes;
        assert(!granted_[static_cast<std::size_t>(idx)]);
        granted_[static_cast<std::size_t>(idx)] = true;
        return block;
    }

    /// @brief Validate and un-flag a returned block, park it on the recycle
    ///        queue; the accounting-free core shared by both release paths.
    ///        Caller holds `mutex_`.
    ///
    /// @param block The block being returned.
    void GlobalMemoryManager::returnLocked(char* block) {
        assert(pool_ != nullptr);
        assert(block >= pool_ && block < pool_ + cfg_.poolBytes);
        const std::ptrdiff_t offset = block - pool_;
        assert(offset % cfg_.blockBytes == 0);
        const int64_t idx = offset / cfg_.blockBytes;
        assert(granted_[static_cast<std::size_t>(idx)]
            && "block return on a block that is not currently granted");
        granted_[static_cast<std::size_t>(idx)] = false;
        recycled_.push_back(block);
    }

    /// @brief Bytes per block (the grant unit).
    ///
    /// @return `static_block_bytes` of the consumed config. Asserts
    ///         initialized; lock-free (immutable after `init`).
    int32_t GlobalMemoryManager::blockBytes() const {
        assert(pool_ != nullptr);
        return cfg_.blockBytes;
    }

    /// @brief Bytes per page (the per-LB allocation unit).
    ///
    /// @details
    /// Owned here so every per-LB manager carves identically; the page size
    /// is a property of the hierarchy, not of one LB.
    ///
    /// @return `static_page_bytes` of the consumed config. Asserts
    ///         initialized; lock-free (immutable after `init`).
    int32_t GlobalMemoryManager::pageBytes() const {
        assert(pool_ != nullptr);
        return cfg_.pageBytes;
    }

    /// @brief Total number of blocks the pool holds.
    ///
    /// @return `poolBytes / blockBytes`. Asserts initialized.
    int64_t GlobalMemoryManager::totalBlocks() const {
        assert(pool_ != nullptr);
        return totalBlocks_;
    }

    /// @brief Blocks currently granted (telemetry).
    ///
    /// @return Granted minus returned. Thread-safe.
    int64_t GlobalMemoryManager::blocksInUse() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return blocksInUse_;
    }

    /// @brief High-water mark of simultaneously granted blocks (telemetry
    ///        for sizing the config defaults).
    ///
    /// @return Peak of `blocksInUse()` over the instance lifetime.
    ///         Thread-safe.
    int64_t GlobalMemoryManager::peakBlocksInUse() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return peakBlocksInUse_;
    }

    /// @brief Record a paged container's single -> two-level page-directory
    ///        promotion: bump the promotion count and the peak data-page
    ///        high-water (lock-free atomics, relaxed).
    ///
    /// @param numPages Data pages the container holds at the promotion.
    void GlobalMemoryManager::recordTwoLevelPromotion(int32_t numPages) {
        twoLevelPromotions_.fetch_add(1, std::memory_order_relaxed);
        int32_t prev = peakPagesHeld_.load(std::memory_order_relaxed);
        while (numPages > prev
            && !peakPagesHeld_.compare_exchange_weak(
                   prev, numPages, std::memory_order_relaxed)) {
            // prev reloaded by compare_exchange_weak on failure; retry.
        }
    }

    /// @brief Number of single -> two-level page-directory promotions across all
    ///        paged containers this instance has seen.
    ///
    /// @return The promotion count (lock-free).
    int64_t GlobalMemoryManager::twoLevelPromotions() const {
        return twoLevelPromotions_.load(std::memory_order_relaxed);
    }

    /// @brief Peak data-page count observed at any two-level promotion.
    ///
    /// @return The high-water data-page count; 0 if none promoted (lock-free).
    int32_t GlobalMemoryManager::peakPagesHeld() const {
        return peakPagesHeld_.load(std::memory_order_relaxed);
    }

    /// @brief Print the two-level page-directory telemetry to the common
    ///        diagnostics log — the end-of-batch spill summary.
    void GlobalMemoryManager::reportPageStats() const {
        diagnosticsLog() << "[mem] two-level page-directory promotions: "
                  << twoLevelPromotions()
                  << ", peak data pages at a promotion: "
                  << peakPagesHeld() << "\n";
    }

    /// @brief The process-wide static-memory manager instance.
    ///
    /// @details
    /// Production binding for `Memory` / per-LB managers. Unit tests build
    /// private `GlobalMemoryManager` instances instead and leave this one
    /// to the harness-level init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initStaticMemory` before any block traffic).
    GlobalMemoryManager& staticMemory() {
        static GlobalMemoryManager instance;
        return instance;
    }

    /// @brief Initialize the process-wide manager from a config triple.
    ///
    /// @details
    /// Thin forwarder to `staticMemory().init(cfg)`; carries the same
    /// idempotent-same-config / assert-on-mismatch contract. Called by the
    /// `ExpressionAnalyzer` constructor right after the JSON parameter
    /// parse, and by the unit-test harness with a small test pool.
    ///
    /// @param cfg Pool / block / page sizing triple.
    void initStaticMemory(const StaticMemoryConfig& cfg) {
        staticMemory().init(cfg);
    }

    /// @brief The process-wide PERSISTENT static-memory manager instance — the
    ///        second pool that backs `Memory::intToBeProved`.
    ///
    /// @details
    /// A separate reservation from `staticMemory()`, never deloaded, reclaimed
    /// per LB only at discharge. Production binding for `Memory::persistentArena`.
    /// Unit tests build private instances and leave this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initPersistentMemory` before any persistent block traffic).
    GlobalMemoryManager& persistentMemory() {
        static GlobalMemoryManager instance;
        return instance;
    }

    /// @brief Initialize the process-wide persistent manager from a config
    ///        triple (must carry `kind == PoolKind::Persistent`).
    ///
    /// @details
    /// Thin forwarder to `persistentMemory().init(cfg)`; same idempotent-same-
    /// config / assert-on-mismatch contract as `initStaticMemory`. Asserts the
    /// config carries `PoolKind::Persistent` so the exhaustion assert names the
    /// right knob.
    ///
    /// @param cfg Persistent pool / block / page sizing triple.
    void initPersistentMemory(const StaticMemoryConfig& cfg) {
        assert(cfg.kind == PoolKind::Persistent
            && "initPersistentMemory needs a config with kind == Persistent");
        persistentMemory().init(cfg);
    }

    /// @brief The process-wide MAIL static-memory manager instance — the third
    ///        pool that backs the cross-LB pull-model mail log.
    ///
    /// @details
    /// A separate reservation from both `staticMemory()` and
    /// `persistentMemory()`, never deloaded; nothing reads its grant ledger, so
    /// it plays no role in any deload / throttle / steward decision. Production
    /// binding for the `ExpressionAnalyzer`-owned `mailArena`. Unit tests build
    /// private instances and leave this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initMailMemory` before any mail block traffic).
    GlobalMemoryManager& mailMemory() {
        static GlobalMemoryManager instance;
        return instance;
    }

    /// @brief Initialize the process-wide mail manager from a config triple
    ///        (must carry `kind == PoolKind::Mail`).
    ///
    /// @details
    /// Thin forwarder to `mailMemory().init(cfg)`; same idempotent-same-config
    /// / assert-on-mismatch contract as `initStaticMemory`. Asserts the config
    /// carries `PoolKind::Mail` so the exhaustion assert names the right knob.
    ///
    /// @param cfg Mail pool / block / page sizing triple.
    void initMailMemory(const StaticMemoryConfig& cfg) {
        assert(cfg.kind == PoolKind::Mail
            && "initMailMemory needs a config with kind == Mail");
        mailMemory().init(cfg);
    }

    /// @brief The process-wide LB-body static-memory manager instance — the
    ///        fourth pool that backs the LB object store (the `Memory` shells).
    ///
    /// @details
    /// A separate reservation from `staticMemory()`, `persistentMemory()`, and
    /// `mailMemory()`, never deloaded; nothing reads its grant ledger, so it
    /// plays no role in any deload / throttle / steward decision. Production
    /// binding for the `ExpressionAnalyzer`-owned `LbStore`. Unit tests build
    /// private instances and leave this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initLbMemory` before any LB-store block traffic).
    GlobalMemoryManager& lbMemory() {
        static GlobalMemoryManager instance;
        return instance;
    }

    /// @brief Initialize the process-wide LB-body manager from a config triple
    ///        (must carry `kind == PoolKind::Lb`).
    ///
    /// @details
    /// Thin forwarder to `lbMemory().init(cfg)`; same idempotent-same-config
    /// / assert-on-mismatch contract as `initStaticMemory`. Asserts the config
    /// carries `PoolKind::Lb` so the exhaustion assert names the right knob.
    ///
    /// @param cfg LB-body pool / block / page sizing triple.
    void initLbMemory(const StaticMemoryConfig& cfg) {
        assert(cfg.kind == PoolKind::Lb
            && "initLbMemory needs a config with kind == Lb");
        lbMemory().init(cfg);
    }

}
