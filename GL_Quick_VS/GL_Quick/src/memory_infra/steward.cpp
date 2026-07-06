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

#include "steward.hpp"

#include "../memory.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>

namespace gl {

    /// @brief Has a worker been parked on a claim past the stuck deadline?
    ///
    /// @details
    /// Pure predicate over a steady-clock start stamp: returns `true` once
    /// `kStuckSeconds` have elapsed since `start`. A worker stamps `start`
    /// when it first parks on a `Busy` claim and calls this on each spin; a
    /// `true` result makes it print the stuck LB's full chain and assert
    /// (Rule 19 — a genuine hang surfaced at its origin, never a silent
    /// freeze). Reads the monotonic clock only — being a safety tripwire, the
    /// clock read introduces no proof nondeterminism.
    ///
    /// @param start The steady-clock stamp captured when the wait began.
    /// @return `true` when at least `kStuckSeconds` have elapsed since
    ///         `start`, else `false`.
    bool steward::stuckDeadlineExceeded(
        std::chrono::steady_clock::time_point start) {
        return std::chrono::steady_clock::now() - start
               >= std::chrono::seconds(steward::kStuckSeconds);
    }

    /// @brief Stops the thread if still running (defined teardown path
    ///        for scope guards); asserts the quiesce contract through
    ///        `stop()`.
    MemorySteward::~MemorySteward() {
        if (running_) stop();
    }

    /// @brief Launch the custodian thread (parked until woken).
    ///
    /// @details Asserts not already running — one thread per steward,
    ///          one steward per prove scope.
    void MemorySteward::start() {
        assert(!running_ && "MemorySteward::start on a running steward");
        stopRequested_ = false;
        thread_ = std::thread(&MemorySteward::threadMain, this);
        running_ = true;
    }

    /// @brief Stop and join the thread.
    ///
    /// @details
    /// Asserts the steward is quiesced (no pending wake, no drain in
    /// flight) — callers quiesce before stopping; un-woken installed
    /// work is legal here (the kernel still owns its `pendingDischarge`
    /// list, nothing is lost). Defined no-op when never started.
    void MemorySteward::stop() {
        if (!running_) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(!busy_ && !runRequested_
                && "MemorySteward::stop without a preceding quiesce");
            stopRequested_ = true;
        }
        cv_.notify_all();
        thread_.join();
        running_ = false;
        stopRequested_ = false;
    }

    /// @brief Hand the steward the current pending-discharge list
    ///        (kernel barrier, single-threaded).
    ///
    /// @details
    /// Replaces any previously installed, never-woken list. Asserts a
    /// quiesced steward; the drain starts only at a later `wake()`.
    ///
    /// @param lbs       Discharged LBs to dump+release, in the kernel's
    ///                  deterministic enqueue order.
    /// @param directory The deload directory (production: `.deload`).
    void MemorySteward::installDischargeWork(std::vector<Memory*> lbs,
                                             const std::string& directory) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(!busy_ && !runRequested_
            && "installDischargeWork on a non-quiesced steward");
        work_ = std::move(lbs);
        directory_ = directory;
        drainedSinceInstall_ = false;
    }

    /// @brief Hand the steward the kernel-entry eviction plan.
    ///
    /// @details
    /// May land while a barrier-released discharge drain is running —
    /// the two channels are independent; the next pass picks it up.
    /// Asserts the previous eviction list was consumed.
    ///
    /// @param victims   Eviction victims, claims already `Planned`.
    /// @param directory The deload directory (production: `.deload`).
    void MemorySteward::installEvictionWork(std::vector<Memory*> victims,
                                            const std::string& directory) {
        std::lock_guard<std::mutex> lock(mutex_);
        assert(evictionWork_.empty()
            && "installEvictionWork over an unconsumed eviction plan");
        evictionWork_ = std::move(victims);
        directory_ = directory;
    }

    /// @brief Release a drain pass — kernel barrier, kernel entry, or
    ///        the grant trigger's crossing callback.
    ///
    /// @details Asserts installed work of some kind exists; wakes
    ///          coalesce (legal while a pass runs — the loop re-checks
    ///          for work after every pass).
    void MemorySteward::wake() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert((!work_.empty() || !evictionWork_.empty())
                && "MemorySteward::wake without installed work");
            runRequested_ = true;
        }
        cv_.notify_all();
    }

    /// @brief Block until the steward is idle (no requested or in-flight
    ///        drain) — the kernel barrier's first act.
    void MemorySteward::quiesce() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return !runRequested_ && !busy_;
        });
    }

    /// @brief Open the working-set window for a phase sweep (kernel,
    ///        single-threaded, before the phase's pool spawns).
    ///
    /// @param cursor    The phase's dispatch cursor.
    /// @param order     The kernel's active vector.
    /// @param workers   The in-flight margin.
    /// @param directory The deload directory for reloads / deloads.
    void MemorySteward::beginPhaseWindow(
        const std::atomic<std::size_t>* cursor,
        const std::vector<Memory*>* order, unsigned workers,
        const std::string& directory) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(windowCursor_ == nullptr
                && "beginPhaseWindow over an open window");
            assert(cursor != nullptr && order != nullptr);
            windowCursor_ = cursor;
            windowOrder_ = order;
            windowWorkers_ = workers;
            directory_ = directory;
        }
        cv_.notify_all();
    }

    /// @brief Close the working-set window (after a phase join).
    void MemorySteward::endPhaseWindow() {
        std::lock_guard<std::mutex> lock(mutex_);
        windowCursor_ = nullptr;
        windowOrder_ = nullptr;
        windowWorkers_ = 0;
    }

    /// @brief Worker-side eviction: free room for a reload by deloading the
    ///        biggest deloadable LB outside the open window.
    ///
    /// @details
    /// See the header. Snapshots the open window under the mutex, picks the
    /// biggest deloadable LB outside the working set, and CAS-deloads it
    /// (`Idle → Busy → Dumped`). A `false` return means nothing qualified —
    /// the caller reloads anyway, so a genuine pool shortfall surfaces as the
    /// `static_pool_bytes` exhaustion assert, never a silent wait. The chosen
    /// LB and the timing are unobservable (deload is content-invisible).
    bool MemorySteward::evictOneForReload(const std::string& directory) {
        const std::atomic<std::size_t>* cursor;
        const std::vector<Memory*>* order;
        unsigned workers;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            cursor = windowCursor_;
            order = windowOrder_;
            workers = windowWorkers_;
        }
        if (cursor == nullptr || order == nullptr) return false;
        const std::size_t cur = cursor->load(std::memory_order_relaxed);
        const std::size_t margin = static_cast<std::size_t>(workers);
        const std::size_t keepLo = (cur > margin) ? cur - margin : 0;
        const std::size_t keepHi = cur + kLookahead;
        Memory* victim = pickBiggestDeloadable(order, keepLo, keepHi);
        if (victim == nullptr) return false;
        uint8_t expected = static_cast<uint8_t>(Memory::StewardClaim::Idle);
        if (victim->stewardClaim.compare_exchange_strong(
                expected,
                static_cast<uint8_t>(Memory::StewardClaim::Busy),
                std::memory_order_acq_rel)) {
            victim->deloadStaticContainers(directory);
            victim->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Dumped),
                std::memory_order_release);
            return true;
        }
        return false;
    }

    /// @brief The unified per-worker claim+load handshake for all three
    ///        phases (see the header).
    void MemorySteward::claimAndLoadForWork(Memory& lb,
                                            const std::string& directory) {
        const auto waitStart = std::chrono::steady_clock::now();
        for (;;) {
            const uint8_t c =
                lb.stewardClaim.load(std::memory_order_acquire);
            if (c == static_cast<uint8_t>(
                    Memory::StewardClaim::WorkerOwned))
                // A part claimed AND fully loaded it — WorkerOwned is published
                // only after the load completes (below), so a split sibling
                // reading here sees a fully-loaded LB, never a half-rebuilt
                // cold-map index.
                return;
            if (c == static_cast<uint8_t>(Memory::StewardClaim::Idle)
                || c == static_cast<uint8_t>(Memory::StewardClaim::Dumped)) {
                uint8_t exp = c;
                // Claim for LOADING via Busy first — this locks out BOTH the
                // steward and sibling parts while the reload runs; WorkerOwned
                // is published only once the LB is fully loaded. This closes the
                // split-LB race where a sibling read a half-rebuilt cold-map
                // (WorkerOwned was previously set before ensureLoaded ran, and
                // reloadFromImage marks resident() before loadLbMemory fills
                // the containers).
                if (lb.stewardClaim.compare_exchange_strong(exp,
                        static_cast<uint8_t>(Memory::StewardClaim::Busy),
                        std::memory_order_acq_rel)) {
                    if (!lb.lbMemory.manager.resident()) {
                        // Load<->evict exchange: above 1/2 each reload pays
                        // for itself with one deload; above the 3/4 hard bound
                        // keep making room so the reload has blocks (else the
                        // reload's acquireBlock surfaces genuine exhaustion,
                        // never a silent wait).
                        const int64_t total = staticMemory().totalBlocks();
                        if (staticMemory().blocksInUse() * steward::kWakeDen
                                >= total * steward::kWakeNum) {
                            bool evicted = evictOneForReload(directory);
                            while (evicted
                                   && staticMemory().blocksInUse()
                                          * lbdeload::kReleaseHighWaterDen
                                      > total
                                          * lbdeload::kReleaseHighWaterNum)
                                evicted = evictOneForReload(directory);
                        }
                        lb.ensureLoaded(directory);
                    }
                    // Publish ready ONLY now (release): a sibling spinning on
                    // Busy sees WorkerOwned and may read the fully-loaded LB.
                    lb.stewardClaim.store(
                        static_cast<uint8_t>(
                            Memory::StewardClaim::WorkerOwned),
                        std::memory_order_release);
                    return;
                }
                continue;
            }
            // Busy: a sibling part is loading this LB (it will publish
            // WorkerOwned) or the steward is mid-op (it will release to Idle /
            // Dumped); either way wait, bounded by the stuck deadline.
            if (steward::stuckDeadlineExceeded(waitStart)) {
                std::cerr << "STUCK: worker waited >"
                          << steward::kStuckSeconds
                          << "s on a Busy claim for LB "
                          << buildLbChainString(lb) << std::endl;
                assert(false && "worker stuck on a Busy claim");
            }
            std::this_thread::yield();
        }
    }

    /// @brief Select the biggest deloadable LB outside the working set.
    ///
    /// @details
    /// A pure selector (see the header). Walks `order`, skips the kept
    /// window `[keepLo, keepHi)`, and among the resident, active, unsplit,
    /// `Idle` LBs returns the one holding the most blocks (or `nullptr`).
    /// The `Idle` read is a hint — the caller re-establishes it with the
    /// `Idle → Busy` CAS before deloading, so a worker that claims the LB
    /// between this read and that CAS just makes the CAS fail.
    ///
    /// @param order  The kernel's active vector (the processing order).
    /// @param keepLo Inclusive lower bound of the kept working set.
    /// @param keepHi Exclusive upper bound of the kept working set.
    /// @return The biggest deloadable LB outside the window, or `nullptr`.
    Memory* MemorySteward::pickBiggestDeloadable(
        const std::vector<Memory*>* order, std::size_t keepLo,
        std::size_t keepHi) const {
        Memory* best = nullptr;
        int64_t bestHeld = 0;
        for (std::size_t i = 0; i < order->size(); ++i) {
            if (i >= keepLo && i < keepHi) continue;   // working set stays
            Memory* lb = (*order)[i];
            if (!lb->isActive || lb->numberOfParts > 1) continue;
            if (!lb->lbMemory.manager.resident()) continue;
            if (lb->stewardClaim.load(std::memory_order_acquire)
                    != static_cast<uint8_t>(Memory::StewardClaim::Idle))
                continue;
            const int64_t held = lb->lbMemory.manager.blocksHeld();
            if (held > bestHeld) { bestHeld = held; best = lb; }
        }
        return best;
    }

    /// @brief One working-set maintenance pass over the open phase window:
    ///        prefetch the upcoming LBs, drain the rest.
    ///
    /// @details
    /// See the header for the full contract. Prefetch reloads `Dumped` LBs
    /// and compacts fragmented `Idle` ones in the lookahead window; drain
    /// mass-evicts every deloadable LB above the 3/4 hard bound and does the
    /// one-for-one exchange between 1/2 and 3/4. Every touch arbitrates
    /// through the claim word and is content-invisible, so the pass timing —
    /// and whether a worker beats the steward to an LB — cannot shift any
    /// observable.
    void MemorySteward::maintainWorkingSet(
        const std::atomic<std::size_t>* cursor,
        const std::vector<Memory*>* order, unsigned workers,
        const std::string& directory) {
        const std::size_t cur = cursor->load(std::memory_order_relaxed);
        const std::size_t n = order->size();

        // Claim helpers (lambdas local to a covered function — Rule 18
        // exempt). Each arbitrates through the claim word; a worker that
        // wins the LB first makes the CAS fail and the steward skips it.
        auto claimBusy = [](Memory* lb, Memory::StewardClaim from) -> bool {
            uint8_t expected = static_cast<uint8_t>(from);
            return lb->stewardClaim.compare_exchange_strong(
                expected,
                static_cast<uint8_t>(Memory::StewardClaim::Busy),
                std::memory_order_acq_rel);
        };
        auto release = [](Memory* lb, Memory::StewardClaim to) {
            lb->stewardClaim.store(static_cast<uint8_t>(to),
                                   std::memory_order_release);
        };
        auto tryDeload = [&](Memory* lb) {
            if (!lb->isActive || lb->numberOfParts > 1) return;
            if (!lb->lbMemory.manager.resident()) return;
            if (lb->stewardClaim.load(std::memory_order_acquire)
                    != static_cast<uint8_t>(Memory::StewardClaim::Idle))
                return;
            if (claimBusy(lb, Memory::StewardClaim::Idle)) {
                lb->deloadStaticContainers(directory);
                release(lb, Memory::StewardClaim::Dumped);
            }
        };

        // (1) PREFETCH — reload the upcoming LBs (so workers find them
        // resident) and compact a fragmented upcoming one.
        const std::size_t pfFrom = cur + static_cast<std::size_t>(workers);
        const std::size_t pfTo = pfFrom + kLookahead;
        int reloaded = 0;
        for (std::size_t i = pfFrom; i < pfTo && i < n; ++i) {
            Memory* lb = (*order)[i];
            const uint8_t c =
                lb->stewardClaim.load(std::memory_order_acquire);
            if (c == static_cast<uint8_t>(Memory::StewardClaim::Dumped)) {
                if (claimBusy(lb, Memory::StewardClaim::Dumped)) {
                    lb->ensureLoaded(directory);
                    release(lb, Memory::StewardClaim::Idle);
                    ++reloaded;
                }
            }
            else if (c == static_cast<uint8_t>(
                         Memory::StewardClaim::Idle)) {
                const int64_t span = lb->lbMemory.manager.usedBytes();
                const int64_t reclaimable =
                    span - lb->lbMemory.liveBytes();
                if (steward::needsReshuffle(reclaimable, span,
                                            staticMemory().blockBytes())
                    && claimBusy(lb, Memory::StewardClaim::Idle)) {
                    lb->lbMemory.reshuffle(stewardScratch_);
                    release(lb, Memory::StewardClaim::Idle);
                }
            }
        }

        // The working set never deloads: the in-flight margin behind the
        // cursor plus the prefetch margin ahead. Everything else is
        // deloadable.
        const std::size_t margin = static_cast<std::size_t>(workers);
        const std::size_t keepLo = (cur > margin) ? cur - margin : 0;
        const std::size_t keepHi = cur + kLookahead;

        // (2) DRAIN — only over the wake mark; mass-evict over the hard
        // bound, otherwise one-for-one against this pass's reloads.
        const int64_t total = staticMemory().totalBlocks();
        const int64_t used = staticMemory().blocksInUse();
        if (used * lbdeload::kReleaseHighWaterDen
                > total * lbdeload::kReleaseHighWaterNum) {
            for (std::size_t i = 0; i < n; ++i) {
                if (i >= keepLo && i < keepHi) continue;
                tryDeload((*order)[i]);
            }
        }
        else if (used * steward::kWakeDen >= total * steward::kWakeNum) {
            for (int k = 0; k < reloaded; ++k) {
                Memory* victim =
                    pickBiggestDeloadable(order, keepLo, keepHi);
                if (!victim) break;
                tryDeload(victim);
            }
        }
    }

    /// @brief Whether the last installed list has been drained.
    ///
    /// @return `true` after the drain of the most recent install
    ///         completed; `false` after an install with no drain yet.
    bool MemorySteward::drainedSinceInstall() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return drainedSinceInstall_;
    }

    /// @brief Thread body: park on the condition variable, drain the
    ///        installed list on wake, repeat until stop.
    ///
    /// @details
    /// The drain runs outside the steward's mutex — `quiesce()` waiters
    /// see `busy_` until the last LB finished, and the only cross-thread
    /// contact inside the loop is the global manager's mutex on block
    /// returns. Work items are discharged LBs whose content is frozen
    /// (I-112), so the dump bytes are independent
    /// of WHEN this loop runs — asynchronous execution, deterministic
    /// files.
    void MemorySteward::threadMain() {
        for (;;) {
            std::unique_lock<std::mutex> lock(mutex_);
            if (windowCursor_ != nullptr && !stopRequested_
                && !runRequested_) {
                // Working-set mode: poll the phase cursor on a short period.
                // Every window action is content-invisible (reload of an
                // already-written image; deload of frozen-this-pass content;
                // reshuffle), so the poll timing cannot shift any observable.
                cv_.wait_for(lock, std::chrono::microseconds(200));
            }
            else {
                cv_.wait(lock, [this] {
                    return stopRequested_ || runRequested_
                           || windowCursor_ != nullptr;
                });
            }
            if (stopRequested_) {
                assert(!runRequested_
                    && "steward stopped with an unexecuted wake");
                return;
            }
            if (!runRequested_) {
                if (windowCursor_ != nullptr) {
                    const std::atomic<std::size_t>* cursor =
                        windowCursor_;
                    const std::vector<Memory*>* order = windowOrder_;
                    const unsigned workers = windowWorkers_;
                    const std::string directory = directory_;
                    busy_ = true;
                    lock.unlock();
                    maintainWorkingSet(cursor, order, workers, directory);
                    lock.lock();
                    busy_ = false;
                    lock.unlock();
                    cv_.notify_all();
                }
                continue;
            }
            runRequested_ = false;
            busy_ = true;
            std::vector<Memory*> work = std::move(work_);
            work_.clear();
            std::vector<Memory*> evictions = std::move(evictionWork_);
            evictionWork_.clear();
            const std::string directory = directory_;
            lock.unlock();

            // Discharge drain first — frozen content, no contention
            // possible, cheapest relief.
            for (Memory* lb : work)
                lb->deloadStaticContainers(directory);

            // Eviction pass — claim-or-skip (I-122):
            // a failed claim means the victim's phase-1 worker got there
            // first and self-services the dump; both dumpers write the
            // identical barrier-state bytes, so the file set and content
            // stay deterministic, only the dumper identity varies.
            for (Memory* lb : evictions) {
                uint8_t expected = static_cast<uint8_t>(
                    Memory::StewardClaim::Planned);
                if (lb->stewardClaim.compare_exchange_strong(
                        expected,
                        static_cast<uint8_t>(Memory::StewardClaim::Busy),
                        std::memory_order_acq_rel)) {
                    lb->deloadStaticContainers(directory);
                    lb->stewardClaim.store(
                        static_cast<uint8_t>(
                            Memory::StewardClaim::Dumped),
                        std::memory_order_release);
                }
            }

            lock.lock();
            busy_ = false;
            if (!work.empty()) drainedSinceInstall_ = true;
            lock.unlock();
            cv_.notify_all();
        }
    }

}
