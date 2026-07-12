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
#include "deload_stats.hpp"
#include "scratch_arena.hpp"

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

    /// @brief Launch the PLANNER thread (parked until woken) and the I/O
    ///        EXECUTOR pool.
    ///
    /// @details
    /// See the header. Asserts not already running — one planner + one
    /// executor pool per steward, one steward per prove scope. The
    /// per-executor scratch arenas are rebuilt per start so a restarted
    /// steward re-sizes cleanly (the deque never moves an `LbArena`).
    ///
    /// @param workers The kernel worker count (sizes the executor pool);
    ///                0 = no hint, the clamp floor applies.
    void MemorySteward::start(unsigned workers) {
        assert(!running_ && "MemorySteward::start on a running steward");
        stopRequested_ = false;
        const unsigned ioThreads =
            steward::ioThreadCountFor(workers, steward::kIoThreadsOverride);
        ioScratch_.clear();
        for (unsigned e = 0; e < ioThreads; ++e)
            ioScratch_.emplace_back(&lbMemory());
        thread_ = std::thread(&MemorySteward::threadMain, this);
        for (unsigned e = 0; e < ioThreads; ++e)
            ioThreads_.emplace_back(&MemorySteward::ioThreadMain, this, e);
        running_ = true;
    }

    /// @brief Stop and join the planner thread AND the executor pool.
    ///
    /// @details
    /// Asserts the steward is quiesced (no pending wake, no drain in
    /// flight, both I/O rings empty, every executor idle) — callers
    /// quiesce before stopping; un-woken installed work is legal here
    /// (the kernel still owns its `pendingDischarge` list, nothing is
    /// lost). Defined no-op when never started.
    void MemorySteward::stop() {
        if (!running_) return;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(!busy_ && !runRequested_
                && "MemorySteward::stop without a preceding quiesce");
            assert(highCount_ == 0 && lowCount_ == 0 && ioBusy_ == 0
                && "MemorySteward::stop with queued or in-flight I/O "
                   "tasks — quiesce first");
            stopRequested_ = true;
        }
        cv_.notify_all();
        thread_.join();
        for (std::thread& t : ioThreads_) t.join();
        ioThreads_.clear();
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

    /// @brief Block until the steward is FULLY idle — planner parked, both
    ///        I/O rings empty, every executor idle (see the header: the
    ///        rings-empty half makes the barrier's registry read race-free).
    void MemorySteward::quiesce() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] {
            return !runRequested_ && !busy_
                   && highCount_ == 0 && lowCount_ == 0 && ioBusy_ == 0;
        });
    }

    /// @brief Open the working-set window for a phase sweep (kernel,
    ///        single-threaded, before the phase's pool spawns).
    ///
    /// @param phase     The phase id, 1..4 (the telemetry bucket; 4 = barrier seam).
    /// @param cursor    The phase's dispatch cursor.
    /// @param order     The kernel's active vector.
    /// @param workers   The in-flight margin.
    /// @param directory The deload directory for reloads / deloads.
    void MemorySteward::beginPhaseWindow(
        int phase, const std::atomic<std::size_t>* cursor,
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
            // New window identity: tasks planned under an older generation
            // drop at execution (an address compare on pass-local order
            // vectors could alias; the monotone generation cannot).
            ++windowGeneration_;
        }
        // Open the per-phase deload-telemetry window (kernel thread,
        // single-threaded) — snapshots the counters the phase's [DELOAD]
        // line will diff. Pure telemetry, never a steward input.
        deloadStats().beginPhaseWindow(phase);
        cv_.notify_all();
    }

    /// @brief Close the working-set window (after a phase join).
    void MemorySteward::endPhaseWindow() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            windowCursor_ = nullptr;
            windowOrder_ = nullptr;
            windowWorkers_ = 0;
            // Invalidate the closed window's queued tasks: they drop on the
            // generation check at execution (loads bound to the window
            // included; the head-prefetch sentinel opts out).
            ++windowGeneration_;
        }
        // Close the deload-telemetry window: prints the phase's [DELOAD]
        // line when it moved traffic (the return value is telemetry only).
        deloadStats().endPhaseWindow();
    }

    /// @brief Barrier HEAD PREFETCH: one-shot HIGH-lane loads for the head
    ///        of the next iteration's sweep (see the header contract).
    ///
    /// @details
    /// Enqueue-time skips (inactive, non-`Dumped`) are the contract's
    /// defined results; the `Dumped → Busy` CAS at execution remains the
    /// correctness arbiter for anything the skip logic raced. The tasks
    /// carry `kAnyWindowGeneration` so the next phase window's opening does
    /// not invalidate them. The barrier calls this AFTER its quiesce and its
    /// discharge loop, so no queued task can ever observe `dischargedForever`
    /// (set only while the rings are provably empty).
    ///
    /// @param order     The active vector whose head to warm.
    /// @param count     Head slots to warm; clamped to the order size.
    /// @param directory The deload directory the loads read from.
    void MemorySteward::prefetchHead(const std::vector<Memory*>* order,
                                     std::size_t count,
                                     const std::string& directory) {
        assert(order != nullptr);
        const int64_t blockBytes =
            static_cast<int64_t>(staticMemory().blockBytes());
        const int64_t budget =
            reserveBlocks_ / steward::kPrefetchBudgetDivisor;
        int64_t plannedLoadBlocks = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            directory_ = directory;
            // Loads already queued count against the budget (the rings are
            // normally empty here — the barrier quiesced — but the
            // accounting stays uniform with the planner's).
            for (std::size_t k = 0; k < highCount_; ++k) {
                const IoTask& q =
                    highRing_[(highHead_ + k) % steward::kIoRingCapacity];
                if (q.kind == IoTask::Kind::Load)
                    plannedLoadBlocks +=
                        (q.lb->lastRawImageBytes() + blockBytes - 1)
                            / blockBytes;
            }
        }
        const std::size_t n = order->size();
        const std::size_t limit = count < n ? count : n;
        for (std::size_t i = 0; i < limit; ++i) {
            Memory* lb = (*order)[i];
            if (!lb->isActive) continue;   // discharged heads never reload
            if (lb->stewardClaim.load(std::memory_order_acquire)
                    != static_cast<uint8_t>(Memory::StewardClaim::Dumped))
                continue;                  // resident or claimed: no load due
            // PREFETCH BUDGET: these loads execute with no window open — no
            // planner eviction can relieve them and the executor load path
            // has no valve — so the pending total must fit inside the
            // reserve's free blocks. Stopping here is a DEFINED result (the
            // un-warmed tail self-loads through the phase-1 handshake); the
            // unbudgeted burst was the 4 GiB teardown wall.
            const int64_t est =
                (lb->lastRawImageBytes() + blockBytes - 1) / blockBytes;
            if (plannedLoadBlocks + est > budget) break;
            if (enqueueIoTask(IoTask{ IoTask::Kind::Load, lb, i,
                                      steward::kAnyWindowGeneration },
                              /*high=*/true) == IoEnqueue::Queued)
                plannedLoadBlocks += est;
        }
    }

    /// @brief Worker-side eviction: free room by deloading the farthest-
    ///        next-use LB behind the open window's cursor (two-tier Belady).
    ///
    /// @details
    /// See the header. Snapshots the open window under the mutex, computes
    /// the kept range from the CURRENT cursor, picks the victim via the
    /// two-tier Belady selector (`pickVictimTwoTier`), and CAS-deloads it
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
        // Unified working set: in-flight margin behind the cursor plus the
        // prefetch span ahead (the prefetch region's end). Identical to the
        // window maintainWorkingSet keeps, so the two never disagree on which
        // LBs are protected.
        const std::size_t keepLo = (cur > margin) ? cur - margin : 0;
        const std::size_t keepHi =
            cur + margin + steward::kLookaheadWorkerMultiple * margin;
        Memory* victim = pickVictimTwoTier(order, cur, keepLo, keepHi);
        if (victim == nullptr) return false;
        uint8_t expected = static_cast<uint8_t>(Memory::StewardClaim::Idle);
        if (victim->stewardClaim.compare_exchange_strong(
                expected,
                static_cast<uint8_t>(Memory::StewardClaim::Busy),
                std::memory_order_acq_rel)) {
            // Active-LB eviction uses the near-memcpy v4 RAW image
            // (D-195); the worker's ensureLoaded
            // reload dispatches back to raw automatically. Discharge stays
            // canonical v3 (the threadMain work list).
            victim->deloadStaticContainersRaw(directory);
            victim->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Dumped),
                std::memory_order_release);
            return true;
        }
        return false;
    }

    /// @brief The unified per-worker claim+load handshake for all three
    ///        phases (see the header).
    void MemorySteward::claimAndLoadForWork(Memory& lb, int phase,
                                            const std::string& directory) {
        assert(phase >= 1 && phase <= 4
            && "claimAndLoadForWork phase out of range (4 = barrier seam)");
        const auto waitStart = std::chrono::steady_clock::now();
        // Nanoseconds elapsed since a steady-clock stamp (lambda local to a
        // covered function — Rule 18 exempt).
        auto nanosSince = [](std::chrono::steady_clock::time_point t) {
            return std::chrono::duration_cast<std::chrono::nanoseconds>(
                       std::chrono::steady_clock::now() - t).count();
        };
        // WIDENED WORKER PRESSURE VALVE. One counted emergency eviction per
        // claim, at most (no loop — a still-short reload surfaces genuine
        // exhaustion at acquireBlock, Rule 19). The shared executor (lambda
        // local to a covered function — Rule 18 exempt) evicts one
        // behind-cursor victim via evictOneForReload (two-tier Belady) and
        // counts it; the guard flag makes the two triggers below mutually
        // exclusive per claim.
        bool evictedThisClaim = false;
        auto emergencyEvict = [&]() {
            const auto evStart = std::chrono::steady_clock::now();
            if (evictOneForReload(directory)) {
                deloadStats().recordWorkerInlineEvict(nanosSince(evStart));
                deloadStats().emergencyEvictCount.fetch_add(
                    1, std::memory_order_relaxed);
                evictedThisClaim = true;
            }
        };
        // Trigger 1 — the EMERGENCY FLOOR, checked on EVERY claim (resident
        // or cold): the pure-grant path (a worker's task grants scratch /
        // statement blocks with no reload in sight) breaches the reserve just
        // as reloads do, and previously had no pressure valve at all — the
        // post-pager 4 GiB wall. The floor scales with the (test-overridable)
        // reserve so the two never drift apart; production value =
        // steward::kEmergencyFloorBlocks. Pressure reads route through
        // pressurePool() (production: staticMemory()) so a unit test can
        // force the valve with a small injected pool. This live blocksInUse
        // read is a sanctioned pager read (I-114) — content-invisible relief,
        // never a deload-SET decision (I-106).
        {
            const int64_t total = pressurePool().totalBlocks();
            const int64_t freeBlocks = total - pressurePool().blocksInUse();
            if (freeBlocks < reserveBlocks_ / steward::kEmergencyFloorDivisor)
                emergencyEvict();
        }
        // Busy-claim wait is timed only once a wait actually begins (the
        // resident fast path clocks nothing here).
        bool sawBusy = false;
        std::chrono::steady_clock::time_point busyStart;
        for (;;) {
            const uint8_t c =
                lb.stewardClaim.load(std::memory_order_acquire);
            if (c == static_cast<uint8_t>(
                    Memory::StewardClaim::WorkerOwned)) {
                // A part claimed AND fully loaded it — WorkerOwned is published
                // only after the load completes (below), so a split sibling
                // reading here sees a fully-loaded LB, never a half-rebuilt
                // cold-map index. The LB is ready without a self-load: a
                // prefetch hit.
                if (sawBusy)
                    deloadStats().recordWorkerClaimWait(
                        phase, nanosSince(busyStart));
                deloadStats().prefetchHit.fetch_add(
                    1, std::memory_order_relaxed);
                return;
            }
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
                    // Census invariant: Dumped means image-on-disk AND blocks
                    // returned, so a Dumped claim can never cover a resident
                    // arena. Race-free: asserted while THIS thread holds Busy
                    // (the state is frozen). The 4 GiB forensic census caught
                    // 290 Dumped-but-resident LBs from seam reloads that
                    // bypassed the claim word; every reload goes through a
                    // claim door now, so this holds globally.
                    assert((c != static_cast<uint8_t>(
                                Memory::StewardClaim::Dumped)
                            || !lb.lbMemory.manager.resident())
                        && "claim was Dumped but the arena is resident — a "
                           "reload bypassed the claim word");
                    if (sawBusy)
                        deloadStats().recordWorkerClaimWait(
                            phase, nanosSince(busyStart));
                    if (!lb.lbMemory.manager.resident()) {
                        // The prefetch did not land it in time: the worker
                        // self-loads (a prefetch miss).
                        deloadStats().prefetchMiss.fetch_add(
                            1, std::memory_order_relaxed);
                        // Trigger 2 — the RELOAD-FIT check: free blocks that
                        // cannot cover THIS reload (estimated from the LB's
                        // last raw-image byte count) fire the same counted
                        // valve, covering the giant-LB case the emergency
                        // floor alone would miss. Skipped when trigger 1
                        // already evicted this claim (at most one eviction per
                        // claim); if the reload is still short, ensureLoaded's
                        // acquireBlock surfaces genuine exhaustion at its
                        // origin (Rule 19), never a silent wait.
                        if (!evictedThisClaim) {
                            const int64_t total = pressurePool().totalBlocks();
                            const int64_t used = pressurePool().blocksInUse();
                            const int64_t blockBytes =
                                pressurePool().blockBytes();
                            const int64_t needBlocks =
                                (lb.lastRawImageBytes() + blockBytes - 1)
                                    / blockBytes;
                            if (total - used < needBlocks) emergencyEvict();
                        }
                        const auto loadStart =
                            std::chrono::steady_clock::now();
                        lb.ensureLoaded(directory);
                        deloadStats().recordWorkerSelfLoad(
                            nanosSince(loadStart));
                    }
                    else {
                        // Already resident (the steward prefetched it): a hit.
                        deloadStats().prefetchHit.fetch_add(
                            1, std::memory_order_relaxed);
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
            // Dumped); either way wait, bounded by the stuck deadline. Stamp
            // the wait start on the first Busy observation.
            if (!sawBusy) {
                sawBusy = true;
                busyStart = std::chrono::steady_clock::now();
            }
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

    /// @brief Select the farthest-next-use eviction victim BEHIND the cursor.
    ///
    /// @details
    /// A pure selector (see the header). Scans BACKWARD from `keepLo - 1` with
    /// wraparound; the cyclic next-use distance `d(i) = (i - cur) mod n`
    /// strictly decreases along this walk, so the first eligible LB is the
    /// Belady victim. Stops on re-entering `[keepLo, keepHi)` (the whole
    /// out-of-window region covered) or after a full lap. Eligible = active,
    /// unsplit, resident, claim `Idle`, plus the `blocksHeld() >= minBlocks`
    /// size floor.
    ///
    /// @param order     The kernel's active vector.
    /// @param cur       The dispatch cursor (the next-use origin; the window
    ///                  bounds encode it, asserted below).
    /// @param keepLo    Inclusive lower bound of the kept window.
    /// @param keepHi    Exclusive upper bound of the kept window.
    /// @param minBlocks The size floor.
    /// @return The farthest-behind eligible victim, or `nullptr`.
    Memory* MemorySteward::pickVictimBehindCursor(
        const std::vector<Memory*>* order, std::size_t cur,
        std::size_t keepLo, std::size_t keepHi, int64_t minBlocks) const {
        const std::size_t n = order->size();
        if (n == 0) return nullptr;
        // The cursor lies inside its own kept window by construction (keepLo =
        // max(0, cur - workers) <= cur < keepHi = cur + ... ). A violation
        // means a caller passed mismatched bounds (Rule 19 — a contract, not a
        // guard the loop routes around).
        assert(cur >= keepLo && cur < keepHi
            && "pickVictimBehindCursor: cursor outside its own window");
        // Start immediately behind the in-flight margin and walk backward with
        // wraparound. Distance decreases monotonically across the bottom wrap,
        // so the first eligible hit is the farthest-next-use LB.
        std::size_t i = (keepLo == 0) ? (n - 1) : (keepLo - 1);
        for (std::size_t scanned = 0; scanned < n;
             ++scanned, i = (i == 0 ? n - 1 : i - 1)) {
            if (i >= keepLo && i < keepHi) break;   // re-entered the window
            Memory* lb = (*order)[i];
            if (!lb->isActive || lb->numberOfParts > 1) continue;
            if (!lb->lbMemory.manager.resident()) continue;
            if (lb->stewardClaim.load(std::memory_order_acquire)
                    != static_cast<uint8_t>(Memory::StewardClaim::Idle))
                continue;
            if (lb->lbMemory.manager.blocksHeld() < minBlocks) continue;
            return lb;
        }
        return nullptr;
    }

    /// @brief TWO-TIER victim selection: floor-preferred Belady scan, then a
    ///        floorless second scan when nothing above the floor qualifies.
    ///
    /// @details
    /// See the header for the full contract. Tier 1 runs
    /// `pickVictimBehindCursor` with the (test-overridable) size floor
    /// `minEvictBlocks_`; a `nullptr` tier-1 result under a breached reserve
    /// means the evictable population sits BELOW the floor (the early-batch
    /// median-2-block regime), so tier 2 re-runs the same Belady scan with the
    /// floor dropped to one block. Both tiers are DEFINED results (Rule 19):
    /// tier 2 is the survival branch, not a degrade path — an inefficient
    /// tiny-LB eviction beats the pool wall the 4 GiB run died on.
    ///
    /// @param order  The kernel's active vector (the processing order).
    /// @param cur    The dispatch cursor (the next-use origin).
    /// @param keepLo Inclusive lower bound of the kept working set.
    /// @param keepHi Exclusive upper bound of the kept working set.
    /// @return The farthest-behind eligible victim (floor-preferred), or
    ///         `nullptr` when not even a one-block victim exists.
    Memory* MemorySteward::pickVictimTwoTier(
        const std::vector<Memory*>* order, std::size_t cur,
        std::size_t keepLo, std::size_t keepHi) const {
        Memory* victim = pickVictimBehindCursor(order, cur, keepLo, keepHi,
                                                minEvictBlocks_);
        if (victim == nullptr)
            victim = pickVictimBehindCursor(order, cur, keepLo, keepHi, 1);
        return victim;
    }

    /// @brief One PLANNER pass over an open phase window: enqueue prefetch
    ///        loads, reshuffles, and reserve-shortfall evictions onto the
    ///        I/O rings — the executors do the I/O.
    ///
    /// @details
    /// See the header for the full contract. Planning reads claim words as
    /// HINTS only (execution re-arbitrates by CAS), so a stale plan costs a
    /// dropped task, never correctness. The reserve projection counts live
    /// free blocks, in-flight dump blocks (freed-but-unavailable), and the
    /// blocks of eviction tasks already queued, so one shortfall is covered
    /// exactly once across passes.
    void MemorySteward::maintainWorkingSet(
        const std::atomic<std::size_t>* cursor,
        const std::vector<Memory*>* order, unsigned workers) {
        const std::size_t cur = cursor->load(std::memory_order_relaxed);
        const std::size_t n = order->size();
        const int64_t blockBytes =
            static_cast<int64_t>(staticMemory().blockBytes());
        uint64_t gen;
        int64_t pendingEvictBlocks = 0;
        int64_t plannedLoadBlocks = 0;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            gen = windowGeneration_;
            // Blocks of eviction tasks already queued: still resident, still
            // Idle, but about to free — count them so the projection does not
            // enqueue a second wave for the same shortfall.
            for (std::size_t k = 0; k < lowCount_; ++k) {
                const IoTask& q =
                    lowRing_[(lowHead_ + k) % steward::kIoRingCapacity];
                if (q.kind == IoTask::Kind::Evict)
                    pendingEvictBlocks += q.lb->lbMemory.manager.blocksHeld();
            }
            // Estimated blocks of load tasks already queued: they will GRANT
            // when executed — the prefetch budget caps pending + this pass's
            // issues so the loads always fit inside the reserve's free
            // blocks (the load-side twin of the eviction projection).
            for (std::size_t k = 0; k < highCount_; ++k) {
                const IoTask& q =
                    highRing_[(highHead_ + k) % steward::kIoRingCapacity];
                if (q.kind == IoTask::Kind::Load)
                    plannedLoadBlocks +=
                        (q.lb->lastRawImageBytes() + blockBytes - 1)
                            / blockBytes;
            }
        }

        // The unified working set is never planned for eviction: the
        // in-flight margin behind the cursor plus the prefetch span ahead.
        // The prefetch region STARTS AT the cursor and ends where the keep
        // window ends (keepHi), so a freshly prefetched LB is inside the
        // kept set and is never planned as a victim by this same pass.
        const std::size_t margin = static_cast<std::size_t>(workers);
        const std::size_t keepLo = (cur > margin) ? cur - margin : 0;
        const std::size_t keepHi =
            cur + margin + steward::kLookaheadWorkerMultiple * margin;

        // (1) PREFETCH planning — a HIGH-lane load task per Dumped LB in the
        // prefetch region `[cursor, keepHi)`; a LOW-lane reshuffle task per
        // fragmented Idle one. Inactive entries are skipped (the same
        // contract prefetchHead has): an inactive LB never runs again, so
        // prefetching it is always wrong — and the barrier seam window runs
        // over `bodies`, which contains long-discharged LBs whose stale
        // `Dumped` claims a load task would otherwise chase into
        // ensureLoaded's discharged-assert. Phase windows never see inactive
        // entries (their orders are pre-filtered active), so this changes
        // nothing there.
        const int64_t loadBudget =
            reserveBlocks_ / steward::kPrefetchBudgetDivisor;
        for (std::size_t i = cur; i < keepHi && i < n; ++i) {
            Memory* lb = (*order)[i];
            if (!lb->isActive) continue;
            const uint8_t c =
                lb->stewardClaim.load(std::memory_order_acquire);
            if (c == static_cast<uint8_t>(Memory::StewardClaim::Dumped)) {
                // PREFETCH BUDGET: pending + this pass's loads never exceed
                // the budget, so the executor loads always fit inside the
                // reserve's free blocks (a load task has no valve). A
                // budget-stopped load is a DEFINED result — re-planned next
                // pass, or the worker self-loads through the handshake.
                const int64_t est =
                    (lb->lastRawImageBytes() + blockBytes - 1) / blockBytes;
                if (plannedLoadBlocks + est <= loadBudget
                    && enqueueIoTask(IoTask{ IoTask::Kind::Load, lb, i, gen },
                                     /*high=*/true) == IoEnqueue::Queued)
                    plannedLoadBlocks += est;
            }
            else if (c == static_cast<uint8_t>(
                         Memory::StewardClaim::Idle)) {
                const int64_t span = lb->lbMemory.manager.usedBytes();
                const int64_t reclaimable =
                    span - lb->lbMemory.liveBytes();
                if (steward::needsReshuffle(reclaimable, span,
                                            staticMemory().blockBytes()))
                    enqueueIoTask(
                        IoTask{ IoTask::Kind::Reshuffle, lb, i, gen },
                        /*high=*/false);
            }
        }

        // (2) DRAIN planning — enqueue evictions totaling the reserve
        // shortfall. Victim enumeration is the TWO-TIER Belady walk: lap 1
        // honors the size floor, lap 2 (only while still short) is floorless
        // — the survival tier: the early-batch population is mostly below
        // the floor, and a floor-only plan finds no victim while grants
        // continue (the post-pager 4 GiB wall). Walking backward from
        // `keepLo - 1` with wraparound visits strictly decreasing next-use
        // distance, so victims enqueue farthest-first. Ends when the
        // projection meets the reserve, the window is re-entered on both
        // laps (no victim — the DEFINED working-set-fits pause; genuine
        // exhaustion still asserts at acquireBlock, Rule 19), or the ring
        // fills (designed load-shedding: counted, re-planned next pass).
        const GlobalMemoryManager& pool = pressurePool();
        const int64_t total = pool.totalBlocks();
        int64_t projected = total - pool.blocksInUse()
                            + inFlightBlocks() + pendingEvictBlocks;
        if (projected >= reserveBlocks_ || n == 0) return;
        const int64_t floors[2] = { minEvictBlocks_, 1 };
        for (int lap = 0; lap < 2 && projected < reserveBlocks_; ++lap) {
            std::size_t i = (keepLo == 0) ? (n - 1) : (keepLo - 1);
            for (std::size_t scanned = 0;
                 scanned < n && projected < reserveBlocks_;
                 ++scanned, i = (i == 0 ? n - 1 : i - 1)) {
                if (i >= keepLo && i < keepHi) break;   // re-entered the window
                Memory* lb = (*order)[i];
                if (!lb->isActive || lb->numberOfParts > 1) continue;
                if (!lb->lbMemory.manager.resident()) continue;
                if (lb->stewardClaim.load(std::memory_order_acquire)
                        != static_cast<uint8_t>(Memory::StewardClaim::Idle))
                    continue;
                const int64_t held = lb->lbMemory.manager.blocksHeld();
                if (held < floors[lap]) continue;
                const IoEnqueue r = enqueueIoTask(
                    IoTask{ IoTask::Kind::Evict, lb, i, gen },
                    /*high=*/false);
                if (r == IoEnqueue::Queued) projected += held;
                else if (r == IoEnqueue::Dropped) return;   // ring full —
                    // load-shed the rest of this plan; next pass regenerates
                // Duplicate: the queued twin's blocks are already counted in
                // pendingEvictBlocks — walk on.
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

    /// @brief Thread body of the PLANNER: park on the condition variable;
    ///        run a planning pass per poll tick while a window is open;
    ///        drain the installed discharge list on wake; repeat until stop.
    ///
    /// @details
    /// The discharge drain is the planner's ONE remaining I/O duty (I-106
    /// execution unchanged — frozen discharged content, deterministic
    /// files); every window I/O runs on the executor pool via the rings.
    /// The drain runs outside the steward's mutex — `quiesce()` waiters
    /// see `busy_` until the last LB finished, and the only cross-thread
    /// contact inside the loop is the global manager's mutex on block
    /// returns.
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
                    busy_ = true;
                    lock.unlock();
                    // Planning only — the executors do the I/O; the tasks
                    // read the window's registered directory at execution.
                    maintainWorkingSet(cursor, order, workers);
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

    /// @brief Enqueue one task onto a ring: dedup by (kind, lb)
    ///        containment, drop-and-count when full, notify.
    ///
    /// @details
    /// See the header. The dedup is hygiene against the 200 µs planner
    /// re-enqueueing a not-yet-executed task every pass; the claim-word CAS
    /// at execution is the correctness arbiter either way. A full ring
    /// DROPS the enqueue — the designed load-shedding (the planner
    /// regenerates next pass), counted per lane.
    ///
    /// @param task The task to queue.
    /// @param high `true` = HIGH ring, `false` = LOW ring.
    /// @return The enqueue outcome (see `IoEnqueue`).
    MemorySteward::IoEnqueue MemorySteward::enqueueIoTask(const IoTask& task,
                                                          bool high) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            IoTask* ring = high ? highRing_ : lowRing_;
            const std::size_t head = high ? highHead_ : lowHead_;
            const std::size_t count = high ? highCount_ : lowCount_;
            for (std::size_t k = 0; k < count; ++k) {
                const IoTask& q =
                    ring[(head + k) % steward::kIoRingCapacity];
                if (q.kind == task.kind && q.lb == task.lb)
                    return IoEnqueue::Duplicate;
            }
            if (count == steward::kIoRingCapacity) {
                if (high)
                    deloadStats().prefetchDroppedFullRing.fetch_add(
                        1, std::memory_order_relaxed);
                else
                    deloadStats().evictDroppedFullRing.fetch_add(
                        1, std::memory_order_relaxed);
                return IoEnqueue::Dropped;
            }
            ring[(head + count) % steward::kIoRingCapacity] = task;
            if (high) ++highCount_; else ++lowCount_;
        }
        cv_.notify_all();
        return IoEnqueue::Queued;
    }

    /// @brief Pop the next task in lane-priority order. Caller holds
    ///        `mutex_`.
    ///
    /// @details
    /// See the header. HIGH first; LOW first when free blocks sit below
    /// the emergency floor (evictions must not starve behind a reload
    /// burst with no blocks to land in). The pressure read takes the pool
    /// mutex UNDER the steward mutex — the one lock order used everywhere.
    ///
    /// @param out Receives the popped task on success.
    /// @return `true` with `out` filled, `false` when both rings empty.
    bool MemorySteward::popIoTaskLocked(IoTask& out) {
        if (highCount_ == 0 && lowCount_ == 0) return false;
        bool takeHigh;
        if (highCount_ == 0) takeHigh = false;
        else if (lowCount_ == 0) takeHigh = true;
        else {
            const GlobalMemoryManager& pool = pressurePool();
            const int64_t freeBlocks =
                pool.totalBlocks() - pool.blocksInUse();
            const bool lowFirst =
                freeBlocks < reserveBlocks_ / steward::kEmergencyFloorDivisor;
            takeHigh = !lowFirst;
        }
        if (takeHigh) {
            out = highRing_[highHead_];
            highHead_ = (highHead_ + 1) % steward::kIoRingCapacity;
            --highCount_;
        }
        else {
            out = lowRing_[lowHead_];
            lowHead_ = (lowHead_ + 1) % steward::kIoRingCapacity;
            --lowCount_;
        }
        return true;
    }

    /// @brief Execute one popped task: re-validate, claim, do the I/O,
    ///        release.
    ///
    /// @details
    /// See the header for the per-kind contract. Every early return is a
    /// DEFINED drop (a worker or a newer plan owns the LB — the claim-first
    /// drop-on-lose protocol), never a failure. The eligibility pre-checks
    /// before the eviction/reshuffle CAS mirror the planner's: after a
    /// successful `Idle → Busy` CAS no other party can have touched the LB
    /// (workers write only LBs they hold `WorkerOwned`), so no post-CAS
    /// re-check is needed.
    ///
    /// @param task    The task to execute.
    /// @param scratch The executing thread's private compaction scratch.
    void MemorySteward::executeIoTask(const IoTask& task, LbArena& scratch) {
        Memory* lb = task.lb;
        switch (task.kind) {
        case IoTask::Kind::Load: {
            std::string directory;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                // A load bound to a closed or superseded window is pointless
                // churn: drop; the planner re-plans. The head-prefetch
                // sentinel executes regardless of window state.
                if (task.windowGen != steward::kAnyWindowGeneration
                    && (windowCursor_ == nullptr
                        || windowGeneration_ != task.windowGen))
                    return;
                directory = directory_;
            }
            uint8_t expected =
                static_cast<uint8_t>(Memory::StewardClaim::Dumped);
            if (!lb->stewardClaim.compare_exchange_strong(
                    expected,
                    static_cast<uint8_t>(Memory::StewardClaim::Busy),
                    std::memory_order_acq_rel))
                return;   // a worker got there first — defined hand-over
            // Census invariant (see claimAndLoadForWork): a Dumped claim can
            // never cover a resident arena — race-free while holding Busy.
            assert(!lb->lbMemory.manager.resident()
                && "claim was Dumped but the arena is resident — a reload "
                   "bypassed the claim word");
            lb->ensureLoaded(directory);
            lb->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
            deloadStats().prefetchIssued.fetch_add(
                1, std::memory_order_relaxed);
            return;
        }
        case IoTask::Kind::Evict: {
            std::size_t cur = 0;
            unsigned workers = 0;
            std::string directory;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                // Re-validate under the CURRENT cursor: the planning window
                // must still be the open one.
                if (windowCursor_ == nullptr
                    || windowGeneration_ != task.windowGen)
                    return;
                cur = windowCursor_->load(std::memory_order_relaxed);
                workers = windowWorkers_;
                directory = directory_;
            }
            const std::size_t margin = static_cast<std::size_t>(workers);
            const std::size_t keepLo = (cur > margin) ? cur - margin : 0;
            const std::size_t keepHi =
                cur + margin + steward::kLookaheadWorkerMultiple * margin;
            // The cursor advanced since planning and the victim's slot slid
            // into the kept working set: drop (it is about to be needed).
            if (task.idx >= keepLo && task.idx < keepHi) return;
            if (!lb->isActive || lb->numberOfParts > 1) return;
            if (!lb->lbMemory.manager.resident()) return;
            uint8_t expected =
                static_cast<uint8_t>(Memory::StewardClaim::Idle);
            if (!lb->stewardClaim.compare_exchange_strong(
                    expected,
                    static_cast<uint8_t>(Memory::StewardClaim::Busy),
                    std::memory_order_acq_rel))
                return;
            // ASYNC DUMP CONTRACT: the LB stays Busy and its blocks stay
            // held until the raw image is complete on disk; the release
            // happens inside deloadStaticContainersRaw (dump, then release),
            // and only THEN is Dumped published — Dumped is never observable
            // before image-complete + blocks-returned. The held blocks are
            // accounted freed-but-unavailable for the planner's projection.
            const int64_t held = lb->lbMemory.manager.blocksHeld();
            const int64_t heldBytes =
                held * static_cast<int64_t>(staticMemory().blockBytes());
            inFlightDumpBlocks_.fetch_add(held, std::memory_order_relaxed);
            inFlightDumpBytes_.fetch_add(heldBytes,
                                         std::memory_order_relaxed);
            const auto evStart = std::chrono::steady_clock::now();
            lb->deloadStaticContainersRaw(directory);
            deloadStats().recordEviction(
                held,
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - evStart).count());
            inFlightDumpBlocks_.fetch_sub(held, std::memory_order_relaxed);
            inFlightDumpBytes_.fetch_sub(heldBytes,
                                         std::memory_order_relaxed);
            lb->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Dumped),
                std::memory_order_release);
            return;
        }
        case IoTask::Kind::Reshuffle: {
            {
                std::lock_guard<std::mutex> lock(mutex_);
                if (windowCursor_ == nullptr
                    || windowGeneration_ != task.windowGen)
                    return;
            }
            // Re-check the gate against current byte counts (the arena may
            // have been compacted or grown since planning), then claim.
            if (lb->stewardClaim.load(std::memory_order_acquire)
                    != static_cast<uint8_t>(Memory::StewardClaim::Idle))
                return;
            const int64_t span = lb->lbMemory.manager.usedBytes();
            const int64_t reclaimable = span - lb->lbMemory.liveBytes();
            if (!steward::needsReshuffle(reclaimable, span,
                                         staticMemory().blockBytes()))
                return;
            uint8_t expected =
                static_cast<uint8_t>(Memory::StewardClaim::Idle);
            if (!lb->stewardClaim.compare_exchange_strong(
                    expected,
                    static_cast<uint8_t>(Memory::StewardClaim::Busy),
                    std::memory_order_acq_rel))
                return;
            lb->lbMemory.reshuffle(scratch);
            lb->stewardClaim.store(
                static_cast<uint8_t>(Memory::StewardClaim::Idle),
                std::memory_order_release);
            return;
        }
        }
        assert(false && "executeIoTask: unreachable task kind");
    }

    /// @brief Thread body of one I/O EXECUTOR: pop tasks in lane-priority
    ///        order and execute them until stop.
    ///
    /// @details
    /// See the header. The pop happens under the mutex the planner enqueues
    /// under; execution runs with the mutex released (the task touches only
    /// its claimed LB, the pool mutex, and the relaxed telemetry atomics).
    /// `ioBusy_` brackets the execution so `quiesce()` waits for in-flight
    /// tasks, not just empty rings.
    ///
    /// @param execIndex This executor's index (selects its private scratch).
    void MemorySteward::ioThreadMain(unsigned execIndex) {
        LbArena& scratch = ioScratch_[execIndex];
        for (;;) {
            IoTask task;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                cv_.wait(lock, [this] {
                    return stopRequested_ || highCount_ > 0 || lowCount_ > 0;
                });
                if (stopRequested_) return;
                const bool popped = popIoTaskLocked(task);
                // The predicate held under this same lock hold, so a task
                // must exist (a racing sibling re-enters the wait instead).
                assert(popped && "ioThreadMain: predicate held but pop "
                                 "found no task");
                if (!popped) continue;
                ++ioBusy_;
            }
            executeIoTask(task, scratch);
            {
                std::lock_guard<std::mutex> lock(mutex_);
                --ioBusy_;
            }
            cv_.notify_all();
        }
    }

    /// @brief Number of tasks currently queued on one I/O ring — a unit
    ///        test's dedup / drop observation point.
    ///
    /// @param high `true` = HIGH ring, `false` = LOW ring.
    /// @return The ring's current task count.
    std::size_t MemorySteward::ioRingDepthForTest(bool high) const {
        std::lock_guard<std::mutex> lock(mutex_);
        return high ? highCount_ : lowCount_;
    }

    /// @brief Pop and execute exactly ONE I/O task on the calling thread,
    ///        honoring the executor lane-priority rule (see the header).
    ///
    /// @return 0 = both rings empty (nothing ran), 1 = a HIGH task ran,
    ///         2 = a LOW task ran.
    int MemorySteward::runOneIoTaskForTest() {
        IoTask task;
        bool wasHigh = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            const std::size_t highBefore = highCount_;
            if (!popIoTaskLocked(task)) return 0;
            wasHigh = (highCount_ < highBefore);
        }
        executeIoTask(task, stewardScratch_);
        return wasHigh ? 1 : 2;
    }

    /// @brief Execute every queued I/O task on the calling thread (see the
    ///        header) — the synchronous test drain.
    void MemorySteward::drainIoRingsForTest() {
        while (runOneIoTaskForTest() != 0) {
        }
    }

    /// @brief Discard every queued-but-unstarted I/O task on both rings —
    ///        prove()'s teardown load-shedding (see the header contract).
    ///
    /// @details
    /// Zeroing the counts under the mutex IS the discard: the rings hold
    /// plain POD tasks (nothing to destruct), the heads stay wherever they
    /// were, and any executor mid-`popIoTaskLocked` sees empty rings on its
    /// next wait. In-flight tasks (already popped, `ioBusy_` counted) run to
    /// completion; the caller's quiesce waits for them.
    void MemorySteward::discardQueuedIoTasks() {
        std::lock_guard<std::mutex> lock(mutex_);
        highCount_ = 0;
        lowCount_ = 0;
    }

    /// @brief Print the one-shot `[EXHAUSTION]` block census to stderr (see
    ///        the header for the fork it resolves and the mutex contract).
    ///
    /// @details
    /// Buckets the grid by raw claim-word value (the enum is sparse: Idle 0,
    /// Planned 1, Busy 3, Dumped 4, WorkerOwned 5), sums `blocksHeld` per
    /// bucket, extracts the eligible-victim bucket from the Idle+resident
    /// population, sums the two scratch registries slot by slot, and closes
    /// with the `UNATTRIBUTED` residue. No manager call anywhere in this
    /// function — audited: claim loads (atomic), `resident()` /
    /// `blocksHeld()` (plain arena bookkeeping, `PtrDirectory::blocksHeld`
    /// included), `isActive` / `numberOfParts` (plain members),
    /// `inFlightBlocks()` / `inFlightBytes()` (relaxed atomics),
    /// `deloadStats()` (relaxed atomics), `ScratchArenaRegistry::slotCount()`
    /// / `forSlot()` (deque reads).
    ///
    /// @param grid        The prover's LB grid (`prove()`'s `bodies`).
    /// @param steward     The live steward (in-flight dump accounting).
    /// @param blocksInUse Pool blocks granted at the wall (passed in).
    /// @param totalBlocks Pool capacity in blocks (passed in).
    void printExhaustionCensus(const std::vector<Memory*>& grid,
                               const MemorySteward* steward,
                               int64_t blocksInUse,
                               int64_t totalBlocks) {
        assert(steward != nullptr
            && "printExhaustionCensus without a steward");
        std::cerr << "[EXHAUSTION] census over " << grid.size()
                  << " grid LBs (CE-clone LBs and sealed page sets are NOT "
                     "grid-reachable — they land in UNATTRIBUTED)"
                  << std::endl;

        // Per-claim-state buckets, indexed by the raw claim byte (sparse
        // enum; value 2 unused).
        struct Bucket {
            int64_t lbs = 0;
            int64_t blocks = 0;
        };
        Bucket byClaim[6];
        Bucket eligible;   // Idle + resident + active + unsplit + >=1 block
        int64_t gridBlocks = 0;
        for (const Memory* lb : grid) {
            if (lb == nullptr) continue;
            const uint8_t c =
                lb->stewardClaim.load(std::memory_order_acquire);
            const int64_t held =
                lb->lbMemory.manager.resident()
                    ? lb->lbMemory.manager.blocksHeld()
                    : 0;
            gridBlocks += held;
            if (c < 6) {
                ++byClaim[c].lbs;
                byClaim[c].blocks += held;
            }
            if (c == static_cast<uint8_t>(Memory::StewardClaim::Idle)
                && lb->lbMemory.manager.resident()
                && lb->isActive && lb->numberOfParts == 1 && held >= 1) {
                ++eligible.lbs;
                eligible.blocks += held;
            }
        }
        const char* names[6] = { "Idle", "Planned", "(unused)",
                                 "Busy", "Dumped", "WorkerOwned" };
        for (int v = 0; v < 6; ++v) {
            if (v == 2) continue;
            std::cerr << "[EXHAUSTION] claim " << names[v]
                      << ": LBs=" << byClaim[v].lbs
                      << " blocks=" << byClaim[v].blocks << std::endl;
        }
        std::cerr << "[EXHAUSTION] eligible victims (Idle+resident+active+"
                     "unsplit, >=1 block): LBs=" << eligible.lbs
                  << " blocks=" << eligible.blocks
                  << "  <- policy-failure signal if large" << std::endl;

        const int64_t inFlight = steward->inFlightBlocks();
        std::cerr << "[EXHAUSTION] in-flight dump: blocks=" << inFlight
                  << " bytes=" << steward->inFlightBytes() << std::endl;

        // Scratch registries: per-slot arena bookkeeping (racy against live
        // workers — approximate by design).
        int64_t scratchStringBlocks = 0;
        int64_t scratchGenBlocks = 0;
        if (scratchArenas().initialized()) {
            const unsigned slots = scratchArenas().slotCount();
            for (unsigned s = 0; s < slots; ++s)
                scratchStringBlocks +=
                    scratchArenas().forSlot(s).blocksHeld();
        }
        if (genScratchArenas().initialized()) {
            const unsigned slots = genScratchArenas().slotCount();
            for (unsigned s = 0; s < slots; ++s)
                scratchGenBlocks +=
                    genScratchArenas().forSlot(s).blocksHeld();
        }
        std::cerr << "[EXHAUSTION] scratch arenas: string blocks="
                  << scratchStringBlocks << " | request-generation blocks="
                  << scratchGenBlocks << " (racy per-slot reads)"
                  << std::endl;

        std::cerr << "[EXHAUSTION] policy constants: kReserveBlocks="
                  << steward::kReserveBlocks << " emergencyFloor="
                  << steward::kEmergencyFloorBlocks << std::endl;

        DeloadStats& ds = deloadStats();
        std::cerr << "[EXHAUSTION] deload stats (cumulative): "
                  << "emergencyEvicts="
                  << ds.emergencyEvictCount.load(std::memory_order_relaxed)
                  << " pagerEvicts="
                  << ds.evictCount.load(std::memory_order_relaxed)
                  << " ringDropsHigh="
                  << ds.prefetchDroppedFullRing.load(
                         std::memory_order_relaxed)
                  << " ringDropsLow="
                  << ds.evictDroppedFullRing.load(std::memory_order_relaxed)
                  << " prefetchIssued="
                  << ds.prefetchIssued.load(std::memory_order_relaxed)
                  << " hit=" << ds.prefetchHit.load(std::memory_order_relaxed)
                  << " miss="
                  << ds.prefetchMiss.load(std::memory_order_relaxed)
                  << std::endl;

        const int64_t unattributed = blocksInUse - gridBlocks - inFlight
                                     - scratchStringBlocks
                                     - scratchGenBlocks;
        std::cerr << "[EXHAUSTION] totals: blocksInUse=" << blocksInUse
                  << "/" << totalBlocks << " gridBlocks=" << gridBlocks
                  << " UNATTRIBUTED=" << unattributed
                  << " (sealed page sets + CE clones + racy skew)"
                  << "  <- pinned-set signal if large" << std::endl;
    }

}
