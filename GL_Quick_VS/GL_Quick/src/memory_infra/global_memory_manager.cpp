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
#include "address_space.hpp"
#include "../infra/diagnostics_log.hpp"

#include "../parameters.hpp"

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <mutex>
#include <new>
#include <thread>

namespace gl {

    /// @brief Intrusive free-slice directory stored in its own physical block.
    /// @details Only blocks with free slices enter the manager's available list.
    /// @invariant The sealed mutex protects these links and the live count.
    struct SharedSealedBlock {
        GlobalMemoryManager* manager;
        SharedSealedBlock* previous;
        SharedSealedBlock* next;
        SealedSlice* free;
        int32_t live;
    };

    /// @brief Compute shared sealed payload capacity from immutable pool geometry.
    /// @details Headers and slice starts are sixteen-byte aligned. Sixteen slices
    /// fit after the block header; larger allocations use ordinary arena blocks.
    /// @return Positive contiguous capacity of each slice.
    int32_t GlobalMemoryManager::sealedSliceCapacity() const {
        assert(initialized());
        const int32_t header = (sizeof(SharedSealedBlock) + 15) & ~15;
        const int32_t stride = ((blockBytes() - header) / 16) & ~15;
        const int32_t payload = stride - static_cast<int32_t>(sizeof(SealedSlice));
        assert(payload > 0 && sizeof(SealedSlice) % 16 == 0);
        return payload;
    }

    /// @brief Lease one slice without exposing peer output bytes.
    /// @details Allocation links are protected by the sealed mutex. The ordinary
    /// pool mutex is acquired only beneath it, never in the reverse order.
    /// @return Empty exclusive slice with stable payload until release.
    /// @invariant Every backing block is charged once as Sealed storage.
    SealedSlice* GlobalMemoryManager::acquireSealedSlice() {
        std::lock_guard<std::mutex> lock(sealedMutex_);
        const int32_t capacity = sealedSliceCapacity();
        if (sealedAvailable_ == nullptr) {
            char* bytes = acquireBlock(PoolPurpose::Sealed);
            auto* block = new (bytes) SharedSealedBlock{this, nullptr, nullptr, nullptr, 0};
            const int32_t header = (sizeof(SharedSealedBlock) + 15) & ~15;
            const int32_t stride = capacity + static_cast<int32_t>(sizeof(SealedSlice));
            for (int32_t i = 15; i >= 0; --i) {
                auto* slice = new (bytes + header + i * stride) SealedSlice{};
                slice->block = block;
                slice->capacity = capacity;
                slice->next = block->free;
                block->free = slice;
            }
            sealedAvailable_ = block;
        }
        SharedSealedBlock* block = sealedAvailable_;
        SealedSlice* slice = block->free;
        assert(slice != nullptr && !slice->leased && block->manager == this);
        block->free = slice->next;
        ++block->live;
        if (block->free == nullptr) {
            sealedAvailable_ = block->next;
            if (sealedAvailable_ != nullptr) sealedAvailable_->previous = nullptr;
            block->next = nullptr;
        }
        slice->next = nullptr;
        slice->used = 0;
        slice->leased = true;
        return slice;
    }

    /// @brief Return a consumed slice and recycle an empty backing block.
    /// @details The free list is independent of record order. Only the final
    /// slice return releases the ordinary pool block; live peers are untouched.
    /// @param slice Live exclusive slice from this manager.
    /// @return Nothing.
    /// @invariant No address enters proof identity or ordering.
    void GlobalMemoryManager::releaseSealedSlice(SealedSlice* slice) {
        std::lock_guard<std::mutex> lock(sealedMutex_);
        assert(slice != nullptr && slice->leased);
        SharedSealedBlock* block = slice->block;
        assert(block != nullptr && block->manager == this && block->live > 0);
        if (block->free == nullptr) {
            block->previous = nullptr;
            block->next = sealedAvailable_;
            if (sealedAvailable_ != nullptr) sealedAvailable_->previous = block;
            sealedAvailable_ = block;
        }
        slice->leased = false;
        slice->next = block->free;
        block->free = slice;
        --block->live;
        if (block->live == 0) {
            if (block->previous != nullptr) block->previous->next = block->next;
            else { assert(sealedAvailable_ == block); sealedAvailable_ = block->next; }
            if (block->next != nullptr) block->next->previous = block->previous;
            releaseBlock(reinterpret_cast<char*>(block), PoolPurpose::Sealed);
        }
    }

    /// @brief Returns the reservation (test instances only — the
    ///        process-wide instance lives until process exit).
    ///
    /// @details
    /// Deliberately does NOT assert `blocksInUse() == 0`: at process exit
    /// LBs are reclaimed by the OS rather than walked, so the process-wide
    /// instance legitimately dies with blocks outstanding. A manager that was
    /// never initialized holds no reservation.
    GlobalMemoryManager::~GlobalMemoryManager() {
        if (pool_ != nullptr) releaseAddressSpace(pool_, cfg_.poolBytes);
    }

    /// @brief Make the one reservation and carve it into blocks.
    ///
    /// @details
    /// Asserts `isValidStaticMemoryConfig` and `isValidStaticPageConfig` on
    /// the sizing triple. The reservation is aligned to 4096 bytes so full
    /// operating-system-page-sized blocks can be registered for CUDA transfers.
    /// It is address space only: `grantLocked` commits each block when it is
    /// first carved (D-418), so the pool's commit charge
    /// tracks currently backed blocks, not `poolBytes`. Idempotent when
    /// called again with a byte-identical config (run modes may construct
    /// more than one `Prover` per process); asserts on a
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
                && cfg.kind == cfg_.kind
                && "GlobalMemoryManager re-init with a different sizing "
                   "triple — one process, one static-memory shape");
            return;
        }
        cfg_ = cfg;
        totalBlocks_ = cfg.poolBytes / cfg.blockBytes;
        pool_ = reserveAddressSpace(cfg.poolBytes);
        if (cfg.kind == PoolKind::GpuHost) {
            assert(totalBlocks_ <= INT32_MAX && cfg.blockBytes >= 4096);
            const int64_t directoryBytes = totalBlocks_ * sizeof(GpuHostBlock);
            gpuHostDirectoryBlocks_ = (directoryBytes + cfg.blockBytes - 1) / cfg.blockBytes;
            assert(gpuHostDirectoryBlocks_ < totalBlocks_);
            commitAddressSpace(pool_, gpuHostDirectoryBlocks_ * cfg.blockBytes);
            gpuHostDirectory_ = reinterpret_cast<GpuHostBlock*>(pool_);
            for (int64_t i = 0; i < totalBlocks_; ++i)
                ::new (static_cast<void*>(gpuHostDirectory_ + i)) GpuHostBlock{};
            for (int64_t i = 0; i < gpuHostDirectoryBlocks_; ++i) {
                gpuHostDirectory_[i].run = -2;
                gpuHostDirectory_[i].committed = 1;
            }
            gpuHostCommittedBlocks_ = gpuHostDirectoryBlocks_;
            blocksInUse_ = peakBlocksInUse_ = gpuHostDirectoryBlocks_;
            blocksInUseHint_.store(blocksInUse_, std::memory_order_relaxed);
            noteGrantLocked(PoolPurpose::GpuHost, gpuHostDirectoryBlocks_);
        } else {
            recycled_.emplace();
            deloadRegistry_.emplace();
            extentSlabRegistry_.emplace();
            granted_.assign(static_cast<std::size_t>(totalBlocks_), false);
            committed_.assign(static_cast<std::size_t>(totalBlocks_), false);
        }
    }

    /// @brief Return one synchronized commitment-counter snapshot.
    ///
    /// @details Current commitment includes both granted and retained free
    /// blocks. The lifetime peak and cumulative release/recommit totals have
    /// separate denominators and are never read to select inference work.
    /// GPU-host commitment only grows, so current equals peak and its
    /// decommit, recommit and trim totals remain zero.
    ///
    /// @return Current and lifetime commitment counters; all zero before init.
    /// @see trimRecycledBlocks
    PoolCommitment GlobalMemoryManager::commitment() const {
        std::lock_guard<std::mutex> lock(mutex_);
        if (cfg_.kind == PoolKind::GpuHost) {
            const int64_t bytes = gpuHostCommittedBlocks_ * cfg_.blockBytes;
            return { bytes, bytes, 0, 0, 0 };
        }
        return commitment_;
    }

    /// @brief Release excess returned blocks while preserving recycle order.
    ///
    /// @details
    /// The first warmBytes/blockBytes entries of the existing recycle queue
    /// are retained for reuse. Later committed entries are decommitted under
    /// the pool mutex, coalescing adjacent addresses in queue order. Granted
    /// blocks, virtual addresses, carve limits and grant accounting stay fixed.
    /// Already decommitted entries remain reserved until their next grant.
    /// This operation is restricted to the four original pools; GPU-host
    /// backing stays committed for reuse.
    ///
    /// @param warmBytes Nonnegative whole-block reuse reserve in bytes.
    /// @return Number of backing bytes decommitted by this call.
    /// @invariant Only exclusively returned whole-page blocks are decommitted.
    /// @see commitment, grantLocked
    int64_t GlobalMemoryManager::trimRecycledBlocks(int64_t warmBytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(pool_ != nullptr && warmBytes >= 0 && warmBytes % cfg_.blockBytes == 0);
        assert(cfg_.kind != PoolKind::GpuHost
            && "GPU-host commitment is retained for reuse");
        ++commitment_.trimCalls;
        const int64_t retained = warmBytes / cfg_.blockBytes;
        int64_t released = 0;
        char* run = nullptr;
        int64_t runBytes = 0;
        const auto flush = [&] {
            if (runBytes == 0) return;
            decommitAddressSpace(run, runBytes);
            commitment_.currentBytes -= runBytes;
            commitment_.decommittedBytes += runBytes;
            released += runBytes;
            run = nullptr;
            runBytes = 0;
        };
        for (int64_t i = retained; i < static_cast<int64_t>(recycled_->size()); ++i) {
            char* const block = (*recycled_)[static_cast<std::size_t>(i)];
            const auto index = static_cast<std::size_t>((block - pool_) / cfg_.blockBytes);
            assert(!granted_[index]);
            if (!committed_[index]) {
                flush();
                continue;
            }
            if (run != nullptr && block != run + runBytes) flush();
            if (run == nullptr) run = block;
            runBytes += cfg_.blockBytes;
            committed_[index] = false;
        }
        flush();
        assert(commitment_.currentBytes >= blocksInUse_ * cfg_.blockBytes);
        return released;
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
    /// @param purpose The consumer class the block serves (attribution
    ///                only; the return must pass the same purpose).
    /// @return Pointer to the granted block of `blockBytes()` bytes.
    ///         Physical identity must not influence any observable
    ///         (I-107).
    char* GlobalMemoryManager::acquireBlock(PoolPurpose purpose) {
        relievePressureBeforeGrant(1);
        char* block = nullptr;
        std::function<void()> fire;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(pool_ != nullptr);
            block = grantLocked(purpose);
            ++blocksInUse_;
            blocksInUseHint_.store(blocksInUse_, std::memory_order_relaxed);
            if (blocksInUse_ > peakBlocksInUse_)
                peakBlocksInUse_ = blocksInUse_;
            noteGrantLocked(purpose, 1);
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
        assert(deloadRegistry_.has_value());
        const auto it = deloadRegistry_->find(ordinal);
        assert((it == deloadRegistry_->end() || it->second == chain)
            && "deload ordinal bound to two different LB chains");
        (*deloadRegistry_)[ordinal] = chain;
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
        assert(deloadRegistry_.has_value());
        return *deloadRegistry_;
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
        assert(deloadRegistry_.has_value());
        deloadRegistry_->clear();
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
        assert(extentSlabRegistry_.has_value());
        extentSlabRegistry_->clear();
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
        (*extentSlabRegistry_)[ordinal] = slab;
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
        const auto it = extentSlabRegistry_->find(ordinal);
        assert(it != extentSlabRegistry_->end()
            && it->second.offset == offset
            && it->second.classBytes == classBytes
            && "extent slab free does not match the registered slab");
        extentSlabRegistry_->erase(it);
        extentAllocator_.freeSlab(offset, classBytes);
    }

    /// @brief Return a granted block to the recycle queue.
    ///
    /// @details
    /// Thread-safe (mutex). Asserts the pointer lies inside the pool, is
    /// block-aligned, and is currently granted (double release and foreign
    /// pointers are bugs we want at their origin).
    ///
    /// @param block   A pointer previously returned by `acquireBlock`.
    /// @param purpose The purpose the block was granted under.
    void GlobalMemoryManager::releaseBlock(char* block, PoolPurpose purpose) {
        std::lock_guard<std::mutex> lock(mutex_);
        returnLocked(block, purpose);
        --blocksInUse_;
        blocksInUseHint_.store(blocksInUse_, std::memory_order_relaxed);
        const int p = static_cast<int>(purpose);
        assert(p >= 0 && p < kPoolPurposeCount);
        --attribution_.inUse[p];
        assert(attribution_.inUse[p] >= 0
            && "pool purpose accounting: a block returned under a purpose it "
               "was not granted under");
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
    /// @param count   Blocks to grant; >= 1.
    /// @param out     Caller array of at least `count` `char*`; filled.
    /// @param purpose The consumer class the blocks serve (attribution only).
    void GlobalMemoryManager::acquireBlocks(int32_t count, char** out,
                                            PoolPurpose purpose) {
        assert(count >= 1);
        relievePressureBeforeGrant(count);
        std::function<void()> fire;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(pool_ != nullptr);
            for (int32_t i = 0; i < count; ++i)
                out[i] = grantLocked(purpose);
            blocksInUse_ += count;
            blocksInUseHint_.store(blocksInUse_, std::memory_order_relaxed);
            if (blocksInUse_ > peakBlocksInUse_)
                peakBlocksInUse_ = blocksInUse_;
            noteGrantLocked(purpose, count);
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
    /// @param blocks  Array of `count` previously granted pointers.
    /// @param count   Blocks to return; >= 1.
    /// @param purpose The purpose the blocks were granted under.
    void GlobalMemoryManager::releaseBlocks(char* const* blocks,
                                            int32_t count,
                                            PoolPurpose purpose) {
        assert(count >= 1);
        std::lock_guard<std::mutex> lock(mutex_);
        for (int32_t i = 0; i < count; ++i)
            returnLocked(blocks[i], purpose);
        blocksInUse_ -= count;
        blocksInUseHint_.store(blocksInUse_, std::memory_order_relaxed);
        const int p = static_cast<int>(purpose);
        assert(p >= 0 && p < kPoolPurposeCount);
        attribution_.inUse[p] -= count;
        assert(attribution_.inUse[p] >= 0
            && "pool purpose accounting: blocks returned under a purpose they "
               "were not granted under");
    }

    /// @brief Book `count` fresh grants under `purpose` and raise every
    ///        high-water they cross.
    ///
    /// @details
    /// Runs inside the grant's mutex section, after `blocksInUse_` and
    /// `peakBlocksInUse_` were advanced. Raises the purpose's own high-water,
    /// then — when the pool just reached a new lifetime high — copies the whole
    /// current purpose split into `atPeak`, so the peak row describes one real
    /// instant. The same snapshot is taken for the current step's high-water.
    /// A handful of integer compares and, only on a new high, a copy of seven
    /// counters: negligible next to the mutex the grant already pays.
    ///
    /// @param purpose The purpose of the granted blocks; not `Count`.
    /// @param count   Blocks just granted; >= 1.
    void GlobalMemoryManager::noteGrantLocked(PoolPurpose purpose,
                                              int64_t count) {
        const int p = static_cast<int>(purpose);
        assert(p >= 0 && p < kPoolPurposeCount && count >= 1);
        PoolAttribution& a = attribution_;
        a.inUse[p] += count;
        if (a.inUse[p] > a.purposePeak[p]) a.purposePeak[p] = a.inUse[p];
        if (blocksInUse_ > a.peakBlocks) {
            a.peakBlocks = blocksInUse_;
            for (int k = 0; k < kPoolPurposeCount; ++k) a.atPeak[k] = a.inUse[k];
        }
        const int s = static_cast<int>(currentPhase_);
        if (blocksInUse_ > a.phasePeak[s]) {
            a.phasePeak[s] = blocksInUse_;
            for (int k = 0; k < kPoolPurposeCount; ++k)
                a.phaseAtPeak[s][k] = a.inUse[k];
        }
    }

    /// @brief Set the prover step that subsequent grants are attributed to.
    ///
    /// @details
    /// Takes the pool mutex, switches the current step, and lifts the new
    /// step's high-water to the current block count when that is higher, with
    /// the matching purpose split — a step that grants nothing still reports
    /// the level it ran at.
    ///
    /// @param phase The step now current; not `Count`.
    void GlobalMemoryManager::setAttributionPhase(PoolPhase phase) {
        const int s = static_cast<int>(phase);
        assert(s >= 0 && s < kPoolPhaseCount);
        std::lock_guard<std::mutex> lock(mutex_);
        currentPhase_ = phase;
        PoolAttribution& a = attribution_;
        if (blocksInUse_ > a.phasePeak[s]) {
            a.phasePeak[s] = blocksInUse_;
            for (int k = 0; k < kPoolPurposeCount; ++k)
                a.phaseAtPeak[s][k] = a.inUse[k];
        }
    }

    /// @brief Snapshot the pool's purpose and step attribution.
    ///
    /// @return A copy taken under the pool mutex.
    PoolAttribution GlobalMemoryManager::attribution() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return attribution_;
    }

    /// @brief Printable name of a pool purpose.
    ///
    /// @param purpose A real purpose; asserts on `Count`.
    /// @return A static string literal.
    const char* poolPurposeName(PoolPurpose purpose) {
        switch (purpose) {
        case PoolPurpose::Unattributed:  return "unattributed";
        case PoolPurpose::LbState:       return "lb_state";
        case PoolPurpose::ScratchString: return "scratch_string";
        case PoolPurpose::ScratchGen:    return "scratch_gen";
        case PoolPurpose::Sealed:        return "sealed";
        case PoolPurpose::GpuHost:       return "gpu_host";
        case PoolPurpose::LbShell:       return "lb_shell";
        case PoolPurpose::Count:         break;
        }
        assert(false && "poolPurposeName: not a real purpose");
        return nullptr;
    }

    /// @brief Printable name of a prover step.
    ///
    /// @param phase A real step; asserts on `Count`.
    /// @return A static string literal.
    const char* poolPhaseName(PoolPhase phase) {
        switch (phase) {
        case PoolPhase::Outside:          return "outside";
        case PoolPhase::Phase1:           return "phase1";
        case PoolPhase::Phase2Execute:    return "phase2_execute";
        case PoolPhase::Phase2Projection: return "phase2_projection";
        case PoolPhase::Phase2Device:     return "phase2_device";
        case PoolPhase::Phase2Finalize:   return "phase2_finalize";
        case PoolPhase::Phase3:           return "phase3";
        case PoolPhase::Barrier:          return "barrier";
        case PoolPhase::Count:            break;
        }
        assert(false && "poolPhaseName: not a real step");
        return nullptr;
    }

    /// @brief Install the grant-time pressure valve (see the header).
    ///
    /// @param valve Pressure relief to run before a grant below the floor;
    ///              empty clears.
    /// @param floorBlocks Free blocks a grant must leave; ignored when
    ///                    `valve` is empty.
    /// @return Nothing.
    void GlobalMemoryManager::setPressureValve(std::function<void()> valve,
                                               int64_t floorBlocks) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(!valve || floorBlocks >= 1);
        pressureFloorBlocks_.store(valve ? floorBlocks : -1, std::memory_order_relaxed);
        pressureValve_ = std::move(valve);
    }

    /// @brief Run the pressure valve when a grant of `count` blocks would
    ///        leave fewer free blocks than the valve's floor.
    ///
    /// @details
    /// Lock-free check on `blocksInUseHint_`; below the floor, copies the
    /// valve under `mutex_` and calls it outside the mutex, unless this thread
    /// is already inside a valve call (the re-entry is the outer call's own
    /// relief). No valve installed is a defined no-op.
    ///
    /// @param count Blocks the caller is about to grant.
    /// @return Nothing.
    void GlobalMemoryManager::relievePressureBeforeGrant(int64_t count) {
        thread_local bool inValve = false;
        const int64_t floor = pressureFloorBlocks_.load(std::memory_order_relaxed);
        if (floor < 0 || inValve) return;
        const int64_t freeAfter = totalBlocks_
            - blocksInUseHint_.load(std::memory_order_relaxed) - count;
        if (freeAfter >= floor) return;
        std::function<void()> valve;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            valve = pressureValve_;
        }
        if (!valve) return;
        inValve = true;
        valve();
        inValve = false;
    }

    /// @brief Dequeue-or-carve one block and flag it granted; the
    ///        accounting-free core shared by both grant paths. Caller holds
    ///        `mutex_`.
    ///
    /// @details
    /// Recycled blocks are re-granted before fresh ones so the carve cursor
    /// is a high-water mark of simultaneous demand across BOTH grant paths.
    /// A fresh carve or a decommitted recycled block commits before use.
    /// Committed recycled blocks retain their existing backing. Exhaustion — every block carved and none
    /// recycled — is an assert
    /// naming the instance's pool sizing constant (`static_pool_bytes`,
    /// `static_persistent_pool_bytes`, or `static_mail_pool_bytes`, per
    /// `cfg_.kind`); there is no fallback path.
    ///
    /// @param purpose Owning accounting category.
    /// @return The granted block.
    char* GlobalMemoryManager::grantLocked(PoolPurpose purpose) {
        if (cfg_.kind == PoolKind::GpuHost) return grantGpuHostLocked(1, purpose);
        char* block = nullptr;
        const bool recycled = !recycled_->empty();
        if (recycled) {
            block = recycled_->front();
            recycled_->pop_front();
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
                          << " recycled=" << recycled_->size()
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
            case PoolKind::GpuHost:
                assert(false && "GPU-host grants use their own directory");
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
        if (!committed_[static_cast<std::size_t>(idx)]) {
            commitAddressSpace(block, cfg_.blockBytes);
            committed_[static_cast<std::size_t>(idx)] = true;
            commitment_.currentBytes += cfg_.blockBytes;
            commitment_.peakBytes = std::max(commitment_.peakBytes, commitment_.currentBytes);
            if (recycled)
                commitment_.recommittedBytes += cfg_.blockBytes;
        }
        granted_[static_cast<std::size_t>(idx)] = true;
        return block;
    }

    /// @brief Validate and un-flag a returned block, park it on the recycle
    ///        queue; the accounting-free core shared by both release paths.
    ///        Caller holds `mutex_`.
    ///
    /// @param block The block being returned.
    /// @param purpose Original accounting category.
    /// @return Nothing.
    void GlobalMemoryManager::returnLocked(char* block, PoolPurpose purpose) {
        if (cfg_.kind == PoolKind::GpuHost) {
            returnGpuHostLocked(block, 1, purpose);
            return;
        }
        assert(pool_ != nullptr);
        assert(block >= pool_ && block < pool_ + cfg_.poolBytes);
        const std::ptrdiff_t offset = block - pool_;
        assert(offset % cfg_.blockBytes == 0);
        const int64_t idx = offset / cfg_.blockBytes;
        assert(granted_[static_cast<std::size_t>(idx)]
            && "block return on a block that is not currently granted");
        granted_[static_cast<std::size_t>(idx)] = false;
        recycled_->push_back(block);
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
    /// `Prover` constructor right after the JSON parameter
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
    /// binding for the `Prover`-owned `mailArena`. Unit tests build
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
    /// binding for the `Prover`-owned `LbStore`. Unit tests build
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

    /// @brief Allocate a first-fit run while holding the pool mutex.
    /// @details Shares the directory with ordinary arena block grants. Recycled
    /// blocks retain commitment; a newly used range is committed before exposure.
    /// @param count Positive block count.
    /// @param purpose Owning accounting category.
    /// @return Stable committed run; exhaustion asserts before any write.
    char* GlobalMemoryManager::grantGpuHostLocked(int32_t count, PoolPurpose purpose) {
        assert(cfg_.kind == PoolKind::GpuHost && gpuHostDirectory_ != nullptr);
        assert(count > 0 && count <= totalBlocks_);
        assert(static_cast<int>(purpose) >= 0 && static_cast<int>(purpose) < kPoolPurposeCount);
        int64_t start = gpuHostDirectoryBlocks_, free = 0;
        for (int64_t i = start; i < totalBlocks_; ++i) {
            if (gpuHostDirectory_[i].run != 0) {
                if (gpuHostDirectory_[i].run > 0) i += gpuHostDirectory_[i].run - 1;
                start = i + 1;
                free = 0;
            } else if (++free == count) {
                char* address = pool_ + start * cfg_.blockBytes;
                bool needsCommit = false;
                for (int32_t j = 0; j < count; ++j) needsCommit |= !gpuHostDirectory_[start + j].committed;
                if (needsCommit) commitAddressSpace(address, static_cast<int64_t>(count) * cfg_.blockBytes);
                for (int32_t j = 0; j < count; ++j) {
                    auto& entry = gpuHostDirectory_[start + j];
                    gpuHostCommittedBlocks_ += !entry.committed;
                    entry.committed = 1;
                    entry.run = j == 0 ? count : -1;
                    entry.purpose = static_cast<uint8_t>(purpose);
                    assert(!entry.pinned);
                }
                return address;
            }
        }
        std::cerr << "[GPU-HOST-EXHAUSTION] budget_bytes=" << cfg_.poolBytes
            << " requested_bytes=" << static_cast<int64_t>(count) * cfg_.blockBytes
            << " live_bytes=" << blocksInUse_ * cfg_.blockBytes
            << " committed_bytes=" << gpuHostCommittedBlocks_ * cfg_.blockBytes
            << " pinned_bytes=" << gpuHostPinnedBlocks_ * cfg_.blockBytes << std::endl;
        assert(false && "GPU-host pool exhausted: raise GL_GPU_HOST_BUDGET_MIB or static_gpu_host_pool_bytes");
        std::abort();
    }

    /// @brief Validate and free one exact GPU-host lease under the mutex.
    /// @details A partial release or a registered transfer is an ownership bug.
    /// @param address Original lease head.
    /// @param count Original length.
    /// @param purpose Original accounting category.
    /// @return Nothing.
    void GlobalMemoryManager::returnGpuHostLocked(char* address, int32_t count, PoolPurpose purpose) {
        assert(cfg_.kind == PoolKind::GpuHost && count > 0);
        const uintptr_t offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(pool_);
        assert(offset < static_cast<uint64_t>(cfg_.poolBytes) && offset % cfg_.blockBytes == 0);
        const int64_t index = static_cast<int64_t>(offset / cfg_.blockBytes);
        assert(index >= gpuHostDirectoryBlocks_ && count <= totalBlocks_ - index);
        assert(gpuHostDirectory_[index].run == count);
        for (int32_t i = 0; i < count; ++i) {
            auto& entry = gpuHostDirectory_[index + i];
            assert(entry.run == (i == 0 ? count : -1));
            assert(entry.purpose == static_cast<uint8_t>(purpose) && !entry.pinned);
        }
        for (int32_t i = 0; i < count; ++i) gpuHostDirectory_[index + i].run = 0;
    }

    /// @brief Lease a contiguous GPU-host range of whole pool blocks.
    /// @details Directory storage and all consumers share one hard budget.
    /// @param count Positive contiguous block count.
    /// @return Stable address; exhaustion asserts.
    char* GlobalMemoryManager::acquireGpuHostRange(int32_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        char* address = grantGpuHostLocked(count, PoolPurpose::GpuHost);
        blocksInUse_ += count;
        blocksInUseHint_.store(blocksInUse_, std::memory_order_relaxed);
        peakBlocksInUse_ = std::max(peakBlocksInUse_, blocksInUse_);
        grantsSinceBarrier_ += count;
        noteGrantLocked(PoolPurpose::GpuHost, count);
        return address;
    }

    /// @brief Return an exact unregistered GPU-host range.
    /// @details Returns capacity without decommitting or moving other consumers.
    /// @param address Original range head.
    /// @param count Original block count.
    /// @return Nothing.
    void GlobalMemoryManager::releaseGpuHostRange(char* address, int32_t count) {
        std::lock_guard<std::mutex> lock(mutex_);
        returnGpuHostLocked(address, count, PoolPurpose::GpuHost);
        blocksInUse_ -= count;
        blocksInUseHint_.store(blocksInUse_, std::memory_order_relaxed);
        attribution_.inUse[static_cast<int>(PoolPurpose::GpuHost)] -= count;
        assert(blocksInUse_ >= gpuHostDirectoryBlocks_);
    }

    /// @brief Account registration of an exclusive GPU-host range.
    /// @details The exact live lease remains pinned through its transfer event.
    /// @param address Lease head.
    /// @param count Exact lease length.
    /// @param pinned Registration state being entered.
    /// @return Nothing.
    void GlobalMemoryManager::setGpuHostPinned(char* address, int32_t count, bool pinned) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(cfg_.kind == PoolKind::GpuHost && count > 0);
        const uintptr_t offset = reinterpret_cast<uintptr_t>(address) - reinterpret_cast<uintptr_t>(pool_);
        assert(offset < static_cast<uint64_t>(cfg_.poolBytes) && offset % cfg_.blockBytes == 0);
        const int64_t index = static_cast<int64_t>(offset / cfg_.blockBytes);
        assert(index >= gpuHostDirectoryBlocks_ && count <= totalBlocks_ - index);
        assert(gpuHostDirectory_[index].run == count);
        for (int32_t i = 0; i < count; ++i) {
            auto& entry = gpuHostDirectory_[index + i];
            assert(entry.pinned != pinned);
            entry.pinned = pinned;
        }
        gpuHostPinnedBlocks_ += pinned ? count : -count;
        assert(gpuHostPinnedBlocks_ >= 0);
        gpuHostPeakPinnedBlocks_ = std::max(gpuHostPeakPinnedBlocks_, gpuHostPinnedBlocks_);
    }

    /// @brief Measure the largest free GPU-host run without changing state.
    /// @details The mutex makes the sizing snapshot internally consistent.
    /// @return Largest contiguous free block count.
    int64_t GlobalMemoryManager::largestGpuHostRun() const {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(cfg_.kind == PoolKind::GpuHost && gpuHostDirectory_ != nullptr);
        int64_t largest = 0, current = 0;
        for (int64_t i = gpuHostDirectoryBlocks_; i < totalBlocks_; ++i) {
            current = gpuHostDirectory_[i].run == 0 ? current + 1 : 0;
            largest = std::max(largest, current);
        }
        return largest;
    }

    /// @brief Measure committed GPU-host bytes, including recycled blocks.
    /// @details The ownership directory is committed and counted at initialization.
    /// @return Committed bytes under the mutex.
    int64_t GlobalMemoryManager::gpuHostCommittedBytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(cfg_.kind == PoolKind::GpuHost);
        return gpuHostCommittedBlocks_ * cfg_.blockBytes;
    }

    /// @brief Measure live GPU-host registration bytes.
    /// @details Only explicitly registered whole leases contribute.
    /// @return Pinned bytes under the mutex.
    int64_t GlobalMemoryManager::gpuHostPinnedBytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(cfg_.kind == PoolKind::GpuHost);
        return gpuHostPinnedBlocks_ * cfg_.blockBytes;
    }

    /// @brief Return the peak registered GPU-host storage.
    /// @details Reads the block high-water under the ownership mutex.
    /// @return Lifetime maximum pinned bytes.
    int64_t GlobalMemoryManager::gpuHostPeakPinnedBytes() const {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(pool_ != nullptr && cfg_.kind == PoolKind::GpuHost);
        return gpuHostPeakPinnedBlocks_ * cfg_.blockBytes;
    }


    /// @brief Access the separately initialized GPU-host pool.
    /// @details Access alone allocates no backing reservation.
    /// @return Fifth process-wide memory manager.
    GlobalMemoryManager& gpuHostMemory() {
        static GlobalMemoryManager manager;
        return manager;
    }

    /// @brief Initialize the GPU-host manager with immutable geometry.
    /// @details Reinitialization must name the same pool and geometry.
    /// @param cfg Valid positive GPU-host configuration.
    /// @return Nothing.
    void initGpuHostMemory(const StaticMemoryConfig& cfg) {
        assert(cfg.kind == PoolKind::GpuHost);
        gpuHostMemory().init(cfg);
    }

}
