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

#include "lb_deload.hpp"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace gl {

    struct Memory;

    namespace steward {

        /// @brief Steward wake watermark, as a fraction of the pool's
        ///        total blocks: eviction work begins when consumption
        ///        crosses `kWakeNum / kWakeDen` (50%).
        ///
        /// @details
        /// Read against LOGICAL block counts at the kernel's
        /// end-of-iteration barrier, or against the grant ledger
        /// mid-iteration (I-106) — never against
        /// mid-iteration `blocksInUse()`.
        constexpr int64_t kWakeNum = 1;
        constexpr int64_t kWakeDen = 2;

        /// @brief Steward stop watermark: once awake, eviction continues
        ///        until consumption is at or below `kStopNum / kStopDen`
        ///        (40%), then the steward parks.
        ///
        /// @details
        /// The wake/stop gap is the hysteresis band — without it the
        /// steward would oscillate at exactly the wake mark, re-waking
        /// on the first grant after every relief pass (the
        /// kswapd-low/high pattern; D-160).
        constexpr int64_t kStopNum = 2;
        constexpr int64_t kStopDen = 5;

        // The three marks must be strictly ordered: stop < wake <
        // synchronous hard bound (the kernel's kReleaseHighWater path),
        // or the band logic degenerates.
        static_assert(kStopNum * kWakeDen < kWakeNum * kStopDen,
            "steward stop watermark must lie below the wake watermark");
        static_assert(kWakeNum * lbdeload::kReleaseHighWaterDen
                          < lbdeload::kReleaseHighWaterNum * kWakeDen,
            "steward wake watermark must lie below the kernel's "
            "synchronous high-water bound");

        /// @brief Reshuffle gate, fraction part: an LB qualifies only when
        ///        its reclaimable hole bytes are at least `kReshuffleFreeNum /
        ///        kReshuffleFreeDen` (1/8) of its arena span.
        ///
        /// @details
        /// Second half of the dual AND gate (the Redis active-defrag pattern:
        /// an absolute floor AND a percentage must both hold —
        /// D-162). The absolute floor is one block's worth
        /// of bytes (below that, compaction cannot return a single block and
        /// the copies buy nothing).
        constexpr int64_t kReshuffleFreeNum = 1;
        constexpr int64_t kReshuffleFreeDen = 8;

        /// @brief The dual AND reshuffle gate: is this LB fragmented enough
        ///        that the copying compaction pays?
        ///
        /// @details
        /// Both must hold: `reclaimableBytes >= blockBytes` (the absolute
        /// floor — at least one whole block can be returned; below it the
        /// copies buy nothing) AND `reclaimableBytes >= spanBytes / 8` (the
        /// fraction — the holes are a meaningful share of the arena's span).
        /// `reclaimableBytes` is `LbArena::usedBytes()` minus the containers'
        /// live content (`LbMemory::liveBytes()`); `spanBytes` is
        /// `usedBytes()`. Below the gate the defined result is "no reshuffle
        /// needed" — the boundary the design demands, not a fallback. A pure
        /// function of deterministic byte counts; the compaction it gates is
        /// content-invisible, so the decision's timing never affects an
        /// observable.
        ///
        /// @param reclaimableBytes The arena's reclaimable hole bytes.
        /// @param spanBytes        The arena's used span (`usedBytes()`).
        /// @param blockBytes       Bytes per block of the hierarchy config.
        /// @return `true` when compaction is worth running on this LB.
        constexpr bool needsReshuffle(int64_t reclaimableBytes,
                                      int64_t spanBytes,
                                      int64_t blockBytes) {
            return reclaimableBytes >= blockBytes
                && reclaimableBytes * kReshuffleFreeDen
                       >= spanBytes * kReshuffleFreeNum;
        }

        /// @brief Stuck-detection deadline, in seconds: a worker that cannot
        ///        acquire the LB it must process within this long is treated
        ///        as a genuine deadlock and asserts rather than spinning
        ///        forever.
        ///
        /// @details
        /// The working-set pager guarantees forward progress by construction
        /// — every reload either evicts a deloadable LB to make room or
        /// asserts on genuine pool exhaustion — so a worker claim-spin that
        /// outlasts this deadline means an unexpected hang, surfaced loudly at
        /// its origin (Rule 19) instead of the silent freeze the old throttle
        /// gate could produce. It is a SAFETY TRIPWIRE, not a proof input: its
        /// firing depends on wall-clock timing and never influences proof
        /// output (the same status the hashburst diagnostic has). 30 s is far
        /// above any legitimate single-LB load / deload / compaction latency.
        constexpr int64_t kStuckSeconds = 30;

        /// @brief Has a worker been parked on a claim past the stuck deadline?
        ///
        /// @details
        /// Pure predicate over a steady-clock start stamp: returns `true` once
        /// `kStuckSeconds` have elapsed since `start`. A worker stamps `start`
        /// when it first parks on a `Busy` claim and calls this on each spin;
        /// a `true` result makes it print the stuck LB's full chain and
        /// assert. Reads the monotonic clock only — being a safety tripwire,
        /// the clock read introduces no proof nondeterminism.
        ///
        /// @param start The steady-clock stamp captured when the wait began.
        /// @return `true` when at least `kStuckSeconds` have elapsed since
        ///         `start`, else `false`.
        bool stuckDeadlineExceeded(std::chrono::steady_clock::time_point start);

    }

    /// @brief The background memory custodian — a dedicated thread that
    ///        executes barrier-decided deload work off the kernel's
    ///        critical path.
    ///
    /// @details
    /// Decision/execution split (I-106): the kernel
    /// DECIDES at its single-threaded end-of-iteration barrier — what to
    /// install, whether to wake now (over the wake watermark) or to arm
    /// the grant trigger (crossing mid-iteration) — and the steward only
    /// EXECUTES, asynchronously, while the next iteration runs. The work
    /// it touches (discharged LBs, `Memory::deloadStaticContainers`) is
    /// disjoint from everything the phase sweeps read or write: a
    /// discharged LB is out of every active snapshot forever
    /// (I-112), so the only shared state is the
    /// global manager's mutex on block returns.
    ///
    /// Barrier protocol (the caller's obligations, asserted here):
    /// `quiesce()` FIRST — every previously requested drain completes
    /// before any state is read; then fold results / decide; then
    /// `installDischargeWork` + (`wake()` now, or arm the trigger whose
    /// callback is `wake()`). `wake()` is legal only after an install
    /// that has not yet drained. `stop()` requires a quiesced steward.
    ///
    /// Lifecycle: one steward per `prove()` call (the iteration loop),
    /// started before the first kernel and stopped — quiesce + join —
    /// after the last. The CE filter and `destroyGrid` never see a live
    /// steward. An assert on the steward thread aborts the process like
    /// any worker (no exception swallowing, Rule 19).
    ///
    /// @invariant At every kernel barrier, after `quiesce()`, the steward
    ///            is idle and every installed-and-woken list has fully
    ///            drained — quiesced block counts are pure functions of
    ///            logical history.
    /// @see `steward::kWakeNum` / `kStopNum` (the watermark band),
    ///      `GlobalMemoryManager::armGrantTrigger` (the mid-iteration
    ///      wake source), D-159.
    class MemorySteward {
    public:
        MemorySteward() = default;

        /// @brief Stops the thread if still running (defined teardown
        ///        path for scope guards); asserts the quiesce contract
        ///        through `stop()`.
        ~MemorySteward();

        MemorySteward(const MemorySteward&) = delete;
        MemorySteward& operator=(const MemorySteward&) = delete;
        MemorySteward(MemorySteward&&) = delete;
        MemorySteward& operator=(MemorySteward&&) = delete;

        /// @brief Launch the custodian thread (parked until woken).
        ///
        /// @details Asserts not already running — one thread per
        ///          steward, one steward per prove scope.
        void start();

        /// @brief Stop and join the thread.
        ///
        /// @details
        /// Asserts the steward is quiesced (no pending wake, no drain in
        /// flight) — callers quiesce before stopping; un-woken installed
        /// work is legal here (the kernel still owns its
        /// `pendingDischarge` list, nothing is lost). Defined no-op when
        /// never started.
        void stop();

        /// @brief Hand the steward the current pending-discharge list
        ///        (kernel barrier, single-threaded).
        ///
        /// @details
        /// Replaces any previously installed, never-woken list (the
        /// kernel re-installs its grown `pendingDischarge` every barrier
        /// until a drain happens). Asserts a quiesced steward. The
        /// steward does NOT start working — a separate `wake()` (now or
        /// from the grant trigger) releases the drain; install-without-
        /// wake is the below-watermark idle path.
        ///
        /// @param lbs       Discharged LBs to dump+release, in the
        ///                  kernel's deterministic enqueue order; the
        ///                  steward takes its own copy of the pointers.
        /// @param directory The deload directory (production: `.deload`).
        void installDischargeWork(std::vector<Memory*> lbs,
                                  const std::string& directory);

        /// @brief Hand the steward the kernel-entry eviction plan —
        ///        active-LB victims whose claims were just set to
        ///        `Planned` (single-threaded, before the phase-1 pool
        ///        spawns).
        ///
        /// @details
        /// Unlike `installDischargeWork` this may land while a
        /// barrier-released discharge drain is still running (the two
        /// channels are independent); the next drain pass picks it up.
        /// Asserts the previous eviction list was consumed. The steward
        /// claims each victim through its claim word
        /// (I-122): a failed claim means the LB's
        /// phase-1 worker won and self-services — the steward skips it.
        ///
        /// @param victims   Eviction victims, biggest-first in the
        ///                  barrier's deterministic order; every entry's
        ///                  claim is `Planned`.
        /// @param directory The deload directory (production: `.deload`).
        void installEvictionWork(std::vector<Memory*> victims,
                                 const std::string& directory);

        /// @brief Release a drain pass — called by the kernel barrier
        ///        (over the wake watermark), the kernel entry (eviction
        ///        plan), or the grant trigger's crossing callback.
        ///
        /// @details Asserts installed work of some kind exists. Legal
        ///          while a pass is already requested or running — wakes
        ///          coalesce: the loop re-checks for work after every
        ///          pass, so a wake during a drain schedules one more
        ///          pass (the kernel-entry eviction wake can overlap the
        ///          barrier's discharge drain).
        void wake();

        /// @brief Block until the steward is idle (no requested or
        ///        in-flight drain) — the kernel barrier's first act.
        ///
        /// @details Immediate return when already idle (the common
        ///          below-watermark case) — a defined result, not a
        ///          fallback.
        void quiesce();

        /// @brief Whether the last installed list has been drained.
        ///
        /// @details Read by the kernel barrier (post-quiesce) to decide
        ///          between folding the drained list's files into the
        ///          manifest and re-installing the grown list.
        ///
        /// @return `true` after the drain of the most recent install
        ///         completed; `false` after an install with no drain
        ///         yet.
        bool drainedSinceInstall() const;

        /// @brief Number of LBs ahead of the phase cursor the steward
        ///        prefetches (the lookahead window).
        static constexpr std::size_t kLookahead = 4;

        /// @brief Open the working-set window for a phase sweep: the steward
        ///        prefetches the upcoming LBs and drains the rest. All three
        ///        phases use this — the working-set pager.
        ///
        /// @details
        /// While the window is open the steward polls the cursor and,
        /// for indices in `[cursor + workers, cursor + workers +
        /// kLookahead)`, prepares the upcoming LBs: reloads a
        /// steward-dumped eviction victim (`Dumped → Busy → Idle`,
        /// claim-protected) and reshuffles a fragmented one (`Idle →
        /// Busy → Idle`, gated by `needsReshuffle`). Every touch goes
        /// through the claim word; a worker that arrives first owns its
        /// LB (`WorkerOwned`) and the steward's claim fails — the
        /// defined hand-it-over-unprepared result. Registered
        /// single-threaded at kernel entry, before the phase-1 pool
        /// spawns.
        ///
        /// @param cursor    The phase-1 sweep's dispatch cursor (indices
        ///                  below it are dispatched or done).
        /// @param order     The kernel's active vector (outlives the
        ///                  window; the barrier quiesce outlives every
        ///                  steward read of it).
        /// @param workers   Worker count — the in-flight margin between
        ///                  the cursor and the window start.
        /// @param directory The deload directory for prefetch reloads.
        void beginPhaseWindow(const std::atomic<std::size_t>* cursor,
                              const std::vector<Memory*>* order,
                              unsigned workers,
                              const std::string& directory);

        /// @brief Close the working-set window (right after a phase join).
        ///
        /// @details
        /// No quiesce needed: every active LB's handshake ran before the
        /// join and outwaits a `Busy` steward, so the steward cannot
        /// hold a claim on any LB past the join; a scan pass straggling
        /// on the stale window only fails CASes against `WorkerOwned`
        /// claims and stops at the next registration check.
        void endPhaseWindow();

        /// @brief Worker-side eviction: free room for a reload by deloading
        ///        the biggest deloadable LB outside the open window.
        ///
        /// @details
        /// The "backend writes its own victim" path (the user's "if a worker
        /// loads one, deload one biggest"). A worker about to reload a cold
        /// LB calls this when the pool is over the watermark; it reads the
        /// open window under the mutex, picks the biggest deloadable LB
        /// (`pickBiggestDeloadable`) outside the working set, and deloads it
        /// (`Idle → Busy → Dumped`). Returns `true` when it freed one,
        /// `false` when nothing qualifies (no window open, or the working
        /// set alone fills the pool — the caller then reloads and a genuine
        /// shortfall surfaces as the `static_pool_bytes` exhaustion assert,
        /// never a silent wait). Content-invisible, so the choice's timing
        /// never shifts proof output.
        ///
        /// @param directory The deload directory for the eviction dump.
        /// @return `true` if an LB was deloaded, `false` if none qualified.
        bool evictOneForReload(const std::string& directory);

        /// @brief Select the biggest deloadable LB outside the working set.
        ///
        /// @details
        /// Scans `order` and returns the resident, active, unsplit, `Idle`
        /// LB at an index OUTSIDE the working-set range `[keepLo, keepHi)`
        /// that holds the most blocks (`blocksHeld()`), or `nullptr` when
        /// none qualifies. "Deloadable" is exactly the working-set policy's
        /// definition (the user's): not in flight (`Idle`, never
        /// `WorkerOwned` / `Busy` / `Planned`), not a split LB (a split LB
        /// stays resident), still active, still resident, and not in the
        /// kept window. Biggest-first means each deload frees the most
        /// blocks per round-trip. A pure SELECTOR with no side effects — the
        /// caller arbitrates the actual deload through a `Idle → Busy` CAS,
        /// so a worker that claims the chosen LB first simply makes that CAS
        /// fail and the steward moves on (the defined hand-over result). The
        /// returned choice is timing-dependent (the resident set varies
        /// run-to-run), which is sound only because deload/reload is
        /// content-invisible ([I-103]).
        ///
        /// @param order  The kernel's active vector (the processing order).
        /// @param keepLo Inclusive lower bound of the kept working set.
        /// @param keepHi Exclusive upper bound of the kept working set.
        /// @return The biggest deloadable LB outside the window, or
        ///         `nullptr` when none qualifies.
        Memory* pickBiggestDeloadable(const std::vector<Memory*>* order,
                                      std::size_t keepLo,
                                      std::size_t keepHi) const;

        /// @brief The unified per-worker claim+load handshake for all three
        ///        phases (the user's "a worker uploads itself").
        ///
        /// @details
        /// Claims `lb` for the calling worker through the claim word and makes
        /// it resident before the worker reads it. An LB already `WorkerOwned`
        /// (a split sibling) returns immediately; `Idle`/`Dumped` CAS to
        /// `WorkerOwned`, and a cold LB is reloaded — first making room via
        /// the load↔evict exchange (above 1/2 each reload deloads one biggest
        /// deloadable LB; above the 3/4 hard bound it keeps evicting until
        /// there is room, else lets the reload's `acquireBlock` surface
        /// genuine exhaustion). `Busy` means the steward is mid-op on this
        /// exact LB; the worker yield-spins, bounded by `kStuckSeconds` — a
        /// longer wait asserts ([I-113], the forward-progress
        /// backstop). The caller releases the claim (`stewardClaim = Idle`)
        /// when it finishes the LB, making it deloadable again.
        ///
        /// @param lb        The LB the worker is about to process.
        /// @param directory The deload directory for reloads / evictions.
        void claimAndLoadForWork(Memory& lb, const std::string& directory);

    private:
        /// @brief Thread body: park on the condition variable, drain the
        ///        installed list on wake, repeat until stop.
        void threadMain();

        /// @brief One working-set maintenance pass over the open phase
        ///        window: prefetch the upcoming LBs, drain the rest.
        ///
        /// @details
        /// Runs on the steward thread with the mutex released; parameters
        /// are the snapshot taken under the mutex. Two parts. (1) PREFETCH:
        /// for the lookahead indices just past the in-flight margin
        /// (`[cursor + workers, cursor + workers + kLookahead)`) reload a
        /// `Dumped` LB (`Dumped → Busy → Idle`) so its worker finds it
        /// resident, and compact a fragmented `Idle` one (`needsReshuffle`).
        /// (2) DRAIN: above the kernel hard bound (3/4 of the pool)
        /// mass-evict EVERY deloadable LB outside the working set down to it;
        /// between the wake mark (1/2) and the hard bound do the one-for-one
        /// exchange — one biggest deloadable LB per reload performed this
        /// pass. The working set kept resident is
        /// `[cursor - workers, cursor + kLookahead)`: the in-flight margin
        /// plus the prefetch margin. Every touch goes through the claim word
        /// and is content-invisible, so its timing never shifts an observable
        /// ([I-103]). Below 1/2 the pass only
        /// prefetches — no deload — so a batch that never crosses 1/2 sees
        /// zero deload/reload traffic.
        ///
        /// @param cursor    The current phase's dispatch cursor.
        /// @param order     The kernel's active vector.
        /// @param workers   The in-flight margin (worker count).
        /// @param directory The deload directory for reloads / deloads.
        void maintainWorkingSet(const std::atomic<std::size_t>* cursor,
                                const std::vector<Memory*>* order,
                                unsigned workers,
                                const std::string& directory);

        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::thread thread_;
        std::vector<Memory*> work_;
        std::vector<Memory*> evictionWork_;
        std::string directory_;
        // Scratch arena for the steward's reshuffle (LbArena::compactPages): drawn
        // from the never-deloaded LB-body pool, NOT the deloadable pool it
        // compacts; owned by the steward thread (the discharge path has its own,
        // since the two reshuffle paths can run concurrently).
        LbArena stewardScratch_{ &lbMemory() };
        const std::atomic<std::size_t>* windowCursor_ = nullptr;
        const std::vector<Memory*>* windowOrder_ = nullptr;
        unsigned windowWorkers_ = 0;
        bool runRequested_ = false;
        bool busy_ = false;
        bool stopRequested_ = false;
        bool drainedSinceInstall_ = false;
        bool running_ = false;
    };
}
