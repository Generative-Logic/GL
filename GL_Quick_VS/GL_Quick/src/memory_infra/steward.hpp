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
#include <deque>
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

        /// @brief Prefetch / keep-window depth as a multiple of the worker
        ///        count.
        ///
        /// @details
        /// The steward keeps `workers + kLookaheadWorkerMultiple * workers`
        /// LBs resident ahead of the dispatch cursor and prefetches that whole
        /// span (the prefetch region and the keep window share this end, so a
        /// freshly prefetched LB is inside the kept set and never drained by
        /// the same pass). 2× workers (64 at 32 workers, a 96-LB prefetch
        /// region ahead of the cursor) is deep enough that a reload started
        /// when the cursor first sees an LB lands before a worker reaches it,
        /// shallow enough not to pin the whole active vector.
        ///
        /// SIZING LAW: the window width MULTIPLIES the concurrent load-side
        /// block demand — every dumped LB inside the region becomes an
        /// in-flight load (a prefetch task, or a worker self-load when the
        /// cursor wins the race), and in-flight loads GRANT blocks with only
        /// the emergency floor protecting them: the executor lane-priority
        /// flip that lets evictions preempt loads engages only below
        /// `kEmergencyFloorBlocks` (reserve / 4). The window's concurrent
        /// load demand must therefore stay UNDER the emergency floor, or the
        /// flip fires too late to save the pool. Measured at a 4 GiB
        /// exhaustion wall with the multiple raised to 4: the widened
        /// window's load stream (prefetch issues plus self-loads, the
        /// emergency valve firing thousands of times) outran the drain and
        /// exhausted the pool while MASSIVE evictable supply stood present —
        /// the census showed nearly the whole pool held by Idle LBs with
        /// eligible victims by the hundreds: a RATE loss, not victim
        /// starvation. Any future widening must scale the emergency floor
        /// with the window. Late-batch, when per-LB images grow, the
        /// region's block estimate can also exceed the prefetch budget
        /// (`reserveBlocks_ / kPrefetchBudgetDivisor`); the budget then caps
        /// the prefetch volume and the un-warmed remainder self-loads — a
        /// defined outcome, see `kPrefetchBudgetBlocks`. Tuned in the dod
        /// subsession.
        constexpr std::size_t kLookaheadWorkerMultiple = 2;

        /// @brief Eviction size floor, in blocks: an LB holding fewer than this
        ///        is never evicted — it stays an implicit permanent resident.
        ///
        /// @details
        /// A size floor lets tiny LBs stay implicit residents rather than be
        /// evicted for almost no reclaimed room: minimizing the eviction
        /// COUNT matters as much as choosing the right victims, so skip LBs
        /// below 4 blocks (1 MiB at 256 KiB blocks) and let them stay
        /// resident. It is a tier-1 preference, not a hard exclusion:
        /// `pickVictimTwoTier`'s second lap is floorless, so under a breached
        /// reserve the floor never blocks survival; in the early-batch
        /// population (median ~2 blocks, almost entirely below the floor)
        /// tier 1 finds little and the floorless tier 2 carries the drain —
        /// an accepted scan cost. Lowering the floor to the population
        /// median (2) was tried only JOINTLY with a widened prefetch window
        /// whose load-rate failure exhausted the pool, so 2 is untested in
        /// isolation; 4 is the value proven across every surviving 4 GiB
        /// batch. The pinned tiny residents cost a bounded slice of the pool
        /// (hygiene, never the eviction mechanism). Tuned in the dod
        /// subsession.
        constexpr int64_t kMinEvictBlocks = 4;

        /// @brief Reserve target, in blocks: the pager keeps at least this many
        ///        free blocks by evicting behind-cursor victims each pass.
        ///
        /// @details
        /// The steady-state free-block cushion that lets a reload find room
        /// without waiting and keeps behind-cursor victims always available
        /// (the eviction-starvation the old mass-evict-then-1-for-1 policy hit
        /// at 4 GiB, where grants outran frees). Each maintenance pass, while
        /// `freeBlocks < kReserveBlocks` and an eligible farthest-behind victim
        /// exists, one victim is evicted; the pass ends when the reserve is met
        /// OR no victim qualifies (a DEFINED result — the working set fits and
        /// the stream pauses; genuine pool exhaustion still asserts at
        /// `acquireBlock`, Rule 19).
        ///
        /// SIZING LAW: the reserve must cover the largest mid-burst TRANSIENT
        /// grant burst — sealed page sets and CE-clone LB arenas, the
        /// allocations the exhaustion census reports as UNATTRIBUTED (not
        /// reachable from any grid registry). That path grants blocks with no
        /// reload in sight and passes NO eviction valve: the planner drain and
        /// the worker pressure valve both relieve pressure by evicting grid
        /// LBs one victim at a time, and neither can keep pace with a burst's
        /// grant rate — the only protection the burst has is the free headroom
        /// already standing when it starts. Measured at a 4 GiB exhaustion
        /// wall: ~2400 UNATTRIBUTED blocks mid-burst, with the emergency valve
        /// firing thousands of times (`emergencyEvictCount`) yet losing the
        /// race per-claim. A reserve below the burst is therefore fatal no
        /// matter how healthy the eviction stream is: 768 blocks exhausted the
        /// pool; 2048 blocks (512 MiB at 256 KiB blocks, 12.5 % of a 4 GiB
        /// pool) is empirically sufficient across every surviving 4 GiB batch.
        /// The reload churn a deep reserve causes (behind-cursor LBs evicted
        /// and re-touched next sweep) is the price of covering the valveless
        /// burst — never thin the reserve below the measured burst, and note
        /// that widening the prefetch window as the churn lever is bounded by
        /// its own load-rate sizing law (see `kLookaheadWorkerMultiple`).
        /// Note the late-batch LB set grows to ~3.4 GiB
        /// (~13.9 K blocks) — early-batch histograms undersell the
        /// steady-state squeeze. Tuned in the dod subsession.
        constexpr int64_t kReserveBlocks = 2048;

        /// @brief Emergency-floor divisor: the floor is the reserve target
        ///        divided by this.
        ///
        /// @details
        /// One named ratio so the production constant
        /// (`kEmergencyFloorBlocks`) and the per-steward floor derived from a
        /// test-overridden reserve (`reserveBlocks_ / kEmergencyFloorDivisor`)
        /// can never drift apart — a test that shrinks the reserve shrinks the
        /// floor in the same proportion.
        constexpr int64_t kEmergencyFloorDivisor = 2;

        /// @brief Emergency floor, in blocks: below this many free blocks the
        ///        pool is in a genuine squeeze and the last-resort relief
        ///        valves engage.
        ///
        /// @details
        /// `kReserveBlocks / kEmergencyFloorDivisor` (1024 blocks = 256 MiB at
        /// 256 KiB blocks). The divisor is 2, not 4: the floor must sit ABOVE
        /// the concurrent in-flight load demand (prefetch loads + worker
        /// self-loads simultaneously granting, measured ~600-1000 blocks at
        /// 4 GiB exhaustion walls where evictable victims stood unused) — the
        /// lane-priority flip below only protects the pool if it engages while
        /// enough free blocks remain to absorb the loads already in flight; at
        /// reserve/4 (512) the flip fired too late and the pool exhausted in a
        /// rate race despite massive eligible-victim supply. Two consumers,
        /// both survival valves rather than steady-state policy:
        /// - the WIDENED WORKER PRESSURE VALVE in `claimAndLoadForWork`: every
        ///   claim — not only a claim whose own reload does not fit — checks
        ///   `freeBlocks` against this floor and, when breached, evicts ONE
        ///   behind-cursor victim (counted, `emergencyEvictCount`). This gives
        ///   the pure-grant path (workers granting scratch / statement blocks
        ///   with no reload in sight) a pressure valve; the post-pager 4 GiB
        ///   failure showed that path had none.
        /// - the I/O executor lane-priority flip: executors serve the HIGH
        ///   (prefetch-load) lane first EXCEPT below this floor, where the LOW
        ///   (eviction) lane goes first — evictions must not starve behind
        ///   reloads when free blocks are nearly gone.
        /// Sits strictly below the reserve target (static_assert), so the
        /// planner's reserve drain always engages before the valves do.
        constexpr int64_t kEmergencyFloorBlocks =
            kReserveBlocks / kEmergencyFloorDivisor;

        static_assert(kEmergencyFloorBlocks >= 1
                          && kEmergencyFloorBlocks < kReserveBlocks,
            "the emergency floor must sit strictly below the reserve target");

        /// @brief Prefetch-budget divisor: the budget is the reserve target
        ///        divided by this.
        ///
        /// @details
        /// One named ratio so the production constant
        /// (`kPrefetchBudgetBlocks`) and the per-steward budget derived from
        /// a test-overridden reserve (`reserveBlocks_ /
        /// kPrefetchBudgetDivisor`) can never drift apart — the same pattern
        /// as `kEmergencyFloorDivisor`.
        constexpr int64_t kPrefetchBudgetDivisor = 2;

        /// @brief Prefetch byte budget, in blocks: pending (queued, not yet
        ///        consumed) prefetch LOADS may never exceed this many
        ///        estimated blocks.
        ///
        /// @details
        /// `kReserveBlocks / kPrefetchBudgetDivisor` (1024 blocks = 256 MiB
        /// at 256 KiB blocks). The load-side twin of the reserve target: the
        /// reserve guarantees free blocks exist; the budget guarantees the
        /// prefetch stream never asks for more than a fraction of them at
        /// once. The prefetch window (`kLookaheadWorkerMultiple`, a 96-LB
        /// region at 32 workers) CAN exceed this budget late-batch, when
        /// per-LB images grow: the budget caps prefetch VOLUME, and the
        /// un-warmed remainder self-loads through the phase-1 handshake — a
        /// defined outcome of the cap, not an error (the window sets the
        /// wish; the budget sets the spend). BUDGET < RESERVE is the
        /// structural survival property — an
        /// executor load task has NO pressure valve (only the worker
        /// handshake does), and at seams with no window open no planner
        /// eviction can relieve it either, so the pending loads must always
        /// fit inside the free blocks the reserve holds: with the budget
        /// enforced, an executor load is structurally unable to hit the
        /// wall. The 4 GiB teardown wall was exactly an unbudgeted burst —
        /// the final barrier's `prefetchHead` enqueued a window-width of
        /// head loads that executed during prove()'s teardown against a
        /// near-full pool. Enforced at BOTH issue sites: `prefetchHead`
        /// (estimated blocks summed per enqueue, stop at the budget) and the
        /// planner's per-pass HIGH-lane enqueues (pending load blocks summed
        /// under the mutex, symmetric to the eviction projection).
        constexpr int64_t kPrefetchBudgetBlocks =
            kReserveBlocks / kPrefetchBudgetDivisor;

        static_assert(kPrefetchBudgetBlocks >= 1
                          && kPrefetchBudgetBlocks < kReserveBlocks,
            "the prefetch budget must sit strictly below the reserve target "
            "- pending loads must always fit inside the reserve's free "
            "blocks");

        /// @brief Capacity of EACH of the two I/O task rings (HIGH and LOW),
        ///        in tasks.
        ///
        /// @details
        /// Fixed at construction — the rings never grow (steward
        /// infrastructure obeys the no-heap-growth discipline). Sizing
        /// rationale: the HIGH lane's worst case is one prefetch region
        /// (`workers + kLookaheadWorkerMultiple * workers` = 96 at 32
        /// workers) plus a barrier head-prefetch batch of the same order —
        /// comfortably under 256. The LOW lane CAN want more than 256
        /// victims in the floorless early-batch regime (a 2048-block deficit
        /// over ~2-block victims ≈ 1024 candidates); the overflow DROP is
        /// the designed load-shedding, not a failure: the planner
        /// regenerates its plan every pass (200 µs), far faster than the
        /// executors consume 256 evictions, so nothing is lost — only
        /// re-planned. Drops are counted (`DeloadStats`
        /// `prefetchDroppedFullRing` / `evictDroppedFullRing`).
        constexpr std::size_t kIoRingCapacity = 256;

        /// @brief Worker-to-executor ratio: one I/O executor per this many
        ///        kernel workers (before clamping).
        constexpr unsigned kIoThreadsPerWorkers = 8;

        /// @brief Minimum I/O executor count (the clamp floor).
        constexpr unsigned kIoThreadsMin = 2;

        /// @brief Maximum I/O executor count (the clamp ceiling).
        constexpr unsigned kIoThreadsMax = 8;

        /// @brief Executor-count override: 0 derives the count from the
        ///        worker count; a nonzero value forces exactly that many
        ///        executors.
        ///
        /// @details
        /// THE CONCURRENCY-ISOLATION SWITCH: set to 1 and rebuild to run
        /// every I/O task on a single executor thread — the pool becomes
        /// serial while the planner/executor split, the rings, and the task
        /// protocol stay exactly as in production, so a suspected
        /// executor-concurrency bug can be isolated without changing any
        /// other behavior. 0 (the production value) selects
        /// `ioThreadCountFor(workers, 0)` = `clamp(workers /
        /// kIoThreadsPerWorkers, kIoThreadsMin, kIoThreadsMax)`.
        constexpr unsigned kIoThreadsOverride = 0;

        /// @brief The I/O executor count for a given kernel worker count.
        ///
        /// @details
        /// `overrideCount` nonzero forces that count (the
        /// `kIoThreadsOverride` isolation switch); otherwise the count is
        /// `workers / kIoThreadsPerWorkers` clamped to `[kIoThreadsMin,
        /// kIoThreadsMax]` — 4 executors at the production 32 workers, and
        /// the clamp floor for a steward started with no worker hint
        /// (`workers == 0`, the unit-test default). Both branches are
        /// defined results of the sizing policy, not fallbacks (Rule 19).
        ///
        /// @param workers       The kernel worker count (0 = no hint).
        /// @param overrideCount 0 = derive from `workers`; nonzero = forced.
        /// @return The number of executor threads to spawn; always >= 1.
        constexpr unsigned ioThreadCountFor(unsigned workers,
                                            unsigned overrideCount) {
            if (overrideCount != 0) return overrideCount;
            const unsigned derived = workers / kIoThreadsPerWorkers;
            if (derived < kIoThreadsMin) return kIoThreadsMin;
            if (derived > kIoThreadsMax) return kIoThreadsMax;
            return derived;
        }

        /// @brief Window-generation sentinel for I/O tasks that are valid
        ///        regardless of which (or whether a) phase window is open.
        ///
        /// @details
        /// Used by the barrier head prefetch, whose loads deliberately cross
        /// the window boundary (no window is open at the barrier). A task
        /// carrying a REAL generation is dropped when its window has closed
        /// (the planner re-plans); a task carrying this sentinel executes
        /// whenever an executor reaches it.
        constexpr uint64_t kAnyWindowGeneration = ~static_cast<uint64_t>(0);

    }

    /// @brief The background memory custodian — a PLANNER thread plus an
    ///        I/O EXECUTOR pool that together execute deload work off the
    ///        kernel's critical path.
    ///
    /// @details
    /// Two-level split. (1) Decision/execution (I-106): the kernel DECIDES
    /// at its single-threaded end-of-iteration barrier — what to install,
    /// whether to wake now (over the wake watermark) or to arm the grant
    /// trigger (crossing mid-iteration) — and the steward only EXECUTES,
    /// asynchronously, while the next iteration runs. (2) Planner/executor:
    /// the planner thread runs the 200 µs window poll and ENQUEUES I/O
    /// tasks (prefetch loads on the HIGH ring; evictions + reshuffles on
    /// the LOW ring) that `steward::ioThreadCountFor` executor threads
    /// drain in lane-priority order — one serial evictor could not match
    /// 32 workers' grant rate against the ~50 ms per-file fixed cost. The
    /// planner's own remaining I/O is exactly the DISCHARGE drain
    /// (`threadMain`'s installed-list branch, I-106 execution unchanged).
    /// Discharged LBs are disjoint from everything the phase sweeps read
    /// or write (I-112); executor tasks on active LBs arbitrate through
    /// the per-LB claim word (I-122), so the only sanctioned shared state
    /// is the global manager's mutex on block traffic (plus the relaxed
    /// `DeloadStats` telemetry atomics).
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

        /// @brief Launch the PLANNER thread (parked until woken) and the I/O
        ///        EXECUTOR pool.
        ///
        /// @details
        /// Asserts not already running — one planner + one executor pool per
        /// steward, one steward per prove scope. The executor count is
        /// `steward::ioThreadCountFor(workers, kIoThreadsOverride)` — 4 at
        /// the production 32 workers, the clamp floor (2) for the no-hint
        /// default a unit test uses, and exactly 1 under the
        /// concurrency-isolation override. Each executor owns a private
        /// compaction scratch arena drawn from the never-deloaded LB-body
        /// pool (`lbMemory()`), so concurrent reshuffle tasks never share
        /// scratch.
        ///
        /// @param workers The kernel worker count (sizes the executor pool);
        ///                0 = no hint, the clamp floor applies.
        void start(unsigned workers = 0);

        /// @brief Stop and join the planner thread AND the executor pool.
        ///
        /// @details
        /// Asserts the steward is quiesced (no pending wake, no drain in
        /// flight, both I/O rings empty, every executor idle) — callers
        /// quiesce before stopping; un-woken installed work is legal here
        /// (the kernel still owns its `pendingDischarge` list, nothing is
        /// lost). Defined no-op when never started.
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

        /// @brief Block until the steward is FULLY idle — no requested or
        ///        in-flight drain, both I/O rings empty, every executor
        ///        idle — the kernel barrier's first act.
        ///
        /// @details
        /// Immediate return when already idle (the common below-watermark
        /// case) — a defined result, not a fallback. The rings-empty +
        /// executors-idle half of the condition is LOAD-BEARING for the
        /// barrier: the barrier reads `GlobalMemoryManager::deloadRegistry()`
        /// by reference (no mutex) for `rewriteRegistry`, and executor dump
        /// tasks WRITE that registry — only a full quiesce makes the
        /// barrier's read race-free. Stale in-ring eviction tasks from a
        /// closed window drop on their generation re-validation, so the
        /// drain-out is quick.
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

        /// @brief Open the working-set window for a phase sweep: the steward
        ///        prefetches the upcoming LBs and drains the rest. All three
        ///        phases use this — the working-set pager.
        ///
        /// @details
        /// While the window is open the steward polls the cursor and, for the
        /// prefetch region `[cursor, cursor + workers +
        /// kLookaheadWorkerMultiple * workers)` (which starts AT the cursor and
        /// ends where the keep window ends), prepares the upcoming LBs: reloads
        /// a steward-dumped eviction victim (`Dumped → Busy → Idle`,
        /// claim-protected) and reshuffles a fragmented one (`Idle →
        /// Busy → Idle`, gated by `needsReshuffle`). Every touch goes
        /// through the claim word; a worker that arrives first owns its
        /// LB (`WorkerOwned`) and the steward's claim fails — the
        /// defined hand-it-over-unprepared result. Registered
        /// single-threaded at kernel entry, before the phase-1 pool
        /// spawns.
        ///
        /// Also opens the deload-telemetry window for this phase
        /// (`DeloadStats::beginPhaseWindow`), which `endPhaseWindow` closes
        /// and reports — pure telemetry, never a steward input.
        ///
        /// @param phase     The phase id, 1..4 (4 = barrier seam) — the telemetry bucket the
        ///                  window's `[DELOAD]` line reports under.
        /// @param cursor    The phase-1 sweep's dispatch cursor (indices
        ///                  below it are dispatched or done).
        /// @param order     The kernel's active vector (outlives the
        ///                  window; the barrier quiesce outlives every
        ///                  steward read of it).
        /// @param workers   Worker count — the in-flight margin between
        ///                  the cursor and the window start.
        /// @param directory The deload directory for prefetch reloads.
        void beginPhaseWindow(int phase,
                              const std::atomic<std::size_t>* cursor,
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
        ///
        /// Closes the deload-telemetry window opened by `beginPhaseWindow`
        /// (`DeloadStats::endPhaseWindow`), which prints the phase's
        /// `[DELOAD]` line when it saw traffic — pure telemetry.
        void endPhaseWindow();

        /// @brief Barrier HEAD PREFETCH: one-shot HIGH-lane loads for the
        ///        head of the next iteration's sweep — kills the phase-1
        ///        cold start every iteration.
        ///
        /// @details
        /// Called by the kernel at the END of its barrier (after the
        /// discharge decision, with no phase window open): enqueues a load
        /// task for every `Dumped`, still-active LB among `order[0,
        /// min(count, n))`. The tasks carry `kAnyWindowGeneration`, so they
        /// execute regardless of window state — deliberately crossing the
        /// iteration boundary (the whole point). Skips are defined results,
        /// not fallbacks (Rule 19): an inactive head entry is contracted to
        /// never reload (it discharged this barrier), a non-`Dumped` one is
        /// resident or claimed and needs no load — calling this with an
        /// all-resident head enqueues nothing (idempotence). Enqueued loads
        /// still arbitrate by the `Dumped → Busy` CAS at execution, so a
        /// worker that reaches the LB first simply wins. Content-invisible:
        /// whether a head LB comes back via this prefetch, the phase-1
        /// window, or a worker self-load never shifts an observable.
        /// BUDGETED: enqueueing stops when the estimated blocks of the
        /// issued loads reach the prefetch budget (`reserveBlocks_ /
        /// kPrefetchBudgetDivisor`) — pending loads must always fit inside
        /// the reserve's free blocks, because these tasks execute with no
        /// window open (no planner eviction can relieve them) and the
        /// executor load path has no valve. Stopping at the budget is a
        /// DEFINED result (the un-warmed tail self-loads through the
        /// phase-1 handshake), not a failure.
        ///
        /// @param order     The active vector whose head to warm (the next
        ///                  iteration re-sweeps it from index 0).
        /// @param count     How many head slots to warm (typically the
        ///                  phase window width); clamped to `order` size.
        /// @param directory The deload directory the loads read from.
        void prefetchHead(const std::vector<Memory*>* order,
                          std::size_t count,
                          const std::string& directory);

        /// @brief Discard every queued-but-unstarted I/O task on both rings
        ///        — prove()'s teardown load-shedding.
        ///
        /// @details
        /// Called by the prove-scope guard BEFORE its quiesce: after the
        /// last iteration, the final barrier's head-prefetch loads (and any
        /// straggler window tasks) are pure waste — there is no next
        /// iteration to warm — and at teardown no window is open, so no
        /// eviction could relieve the loads' grants. Dropping planned-but-
        /// unstarted I/O is DEFINED load-shedding, the same doctrine as the
        /// full-ring enqueue drop (Rule 19: a defined result, not a
        /// fallback); tasks already mid-execution finish normally (bounded
        /// by the executor count) and the guard's quiesce waits for them.
        void discardQueuedIoTasks();

        /// @brief Worker-side eviction: free room by deloading the
        ///        farthest-next-use LB behind the open window's cursor
        ///        (two-tier Belady selection).
        ///
        /// @details
        /// The worker pressure valve's execution arm. Snapshots the open
        /// window under the mutex, computes the kept range from the CURRENT
        /// cursor, picks a victim via the two-tier Belady selector
        /// (`pickVictimTwoTier` — floor-preferred, floorless second scan), and
        /// deloads it as a near-memcpy v4 RAW image (`Idle → Busy → Dumped`,
        /// `D-195`). Returns `true` when it freed one,
        /// `false` when nothing qualifies (no window open, or the working
        /// set alone fills the pool — the caller then reloads and a genuine
        /// shortfall surfaces as the `static_pool_bytes` exhaustion assert,
        /// never a silent wait). Content-invisible, so the choice's timing
        /// never shifts proof output. The former biggest-first selector
        /// (`pickBiggestDeloadable`) is retired: Belady order keeps the
        /// eviction pressure on the LBs whose reload is furthest away, so the
        /// valve never competes with the prefetch window it protects.
        ///
        /// @param directory The deload directory for the eviction dump.
        /// @return `true` if an LB was deloaded, `false` if none qualified.
        bool evictOneForReload(const std::string& directory);

        /// @brief Select the farthest-next-use eviction victim BEHIND the
        ///        cursor — the Belady choice for the reserve-target drain.
        ///
        /// @details
        /// GL processes `order` cyclically (each sweep re-runs 0..n-1), so an
        /// LB's next-use distance from the cursor `c` is `d(i) = (i - c) mod n`.
        /// The largest `d` is the LB immediately behind the in-flight margin;
        /// scanning BACKWARD from `keepLo - 1` with wraparound visits indices in
        /// STRICTLY DECREASING `d`, so the FIRST eligible LB found is the
        /// farthest-next-use victim. The scan stops the moment it re-enters the
        /// kept window `[keepLo, keepHi)` (the whole out-of-window region has
        /// then been covered). Repeatedly calling this while deloading each
        /// returned victim yields the batch of the k farthest-behind eligible
        /// LBs — exactly the LBs whose reload is furthest away, so eviction
        /// pressure lands where it costs the working set least.
        ///
        /// Eligibility: active, unsplit, resident, claim `Idle`, and
        /// `blocksHeld() >= minBlocks` (tiny LBs stay implicit residents —
        /// per-eviction fixed cost makes evicting them net-negative). A pure
        /// SELECTOR, no side effects: the caller re-establishes the claim with
        /// an `Idle → Busy` CAS before deloading, so a worker that grabs the
        /// chosen LB first just makes that CAS fail. The choice is
        /// timing-dependent (the resident set varies run-to-run), sound only
        /// because active-LB eviction/reload is content-invisible ([I-107],
        /// `D-195`).
        ///
        /// @param order    The kernel's active vector (the processing order).
        /// @param cur      The dispatch cursor (the next-use origin).
        /// @param keepLo   Inclusive lower bound of the kept working set.
        /// @param keepHi   Exclusive upper bound of the kept working set (may
        ///                 exceed `order->size()`; treated linearly, so
        ///                 `[keepLo, order->size())` is the effective window).
        /// @param minBlocks The size floor: an LB holding fewer blocks is
        ///                 ineligible.
        /// @return The farthest-next-use eligible victim behind the cursor, or
        ///         `nullptr` when none qualifies (the pass then ends — a
        ///         defined result, the stream pauses).
        Memory* pickVictimBehindCursor(const std::vector<Memory*>* order,
                                       std::size_t cur,
                                       std::size_t keepLo,
                                       std::size_t keepHi,
                                       int64_t minBlocks) const;

        /// @brief TWO-TIER victim selection: the floor-preferred Belady scan,
        ///        then — only when it finds nothing — a floorless second scan.
        ///
        /// @details
        /// Tier 1 is `pickVictimBehindCursor` with the eviction size floor
        /// (`minEvictBlocks_`): the steady-state policy, where evicting a tiny
        /// LB is net-negative (the per-eviction fixed file cost dominates).
        /// Tier 2 re-runs the SAME Belady backward scan with the floor dropped
        /// to one block. It exists because the early-batch population is
        /// mostly BELOW the floor (median ~2 blocks against a 4-block floor),
        /// so under a breached reserve tier 1 can find no victim at all while
        /// grants continue — the post-pager 4 GiB wall. Survival trumps
        /// per-operation efficiency: when the reserve is breached and only
        /// tiny LBs remain evictable, evicting them inefficiently is the
        /// DEFINED result, not a degrade path (Rule 19 — both tiers are
        /// contracted outcomes). Callers invoke this only under a breached
        /// reserve (the drain loop's condition, or the worker valve's
        /// emergency floor which sits below the reserve). A `nullptr` return
        /// means not even a one-block victim exists — the defined
        /// working-set-fits pass end.
        ///
        /// @param order  The kernel's active vector (the processing order).
        /// @param cur    The dispatch cursor (the next-use origin).
        /// @param keepLo Inclusive lower bound of the kept working set.
        /// @param keepHi Exclusive upper bound of the kept working set.
        /// @return The farthest-behind eligible victim (floor-preferred), or
        ///         `nullptr` when not even a one-block victim exists.
        Memory* pickVictimTwoTier(const std::vector<Memory*>* order,
                                  std::size_t cur,
                                  std::size_t keepLo,
                                  std::size_t keepHi) const;

        /// @brief Override the reserve target (`kReserveBlocks`) for a unit
        ///        test; production uses the named constant.
        ///
        /// @details
        /// The reserve-target drain evicts until `freeBlocks >= reserve`. On
        /// the 4 GiB process pool a test cannot create a real 2048-block
        /// deficit, so it lowers or raises the reserve here (typically to
        /// `currentFree + deficit`) to drive a bounded, deterministic eviction
        /// count. The worker valve's emergency floor scales with it
        /// (`reserveBlocks_ / kEmergencyFloorDivisor`), so the floor keeps its
        /// production proportion under an overridden reserve. A pure test
        /// seam — production never calls it, never a proof input (Rule 19:
        /// both the default and the override are defined results, not a
        /// degrade path).
        ///
        /// @param reserve The reserve target in blocks for subsequent passes.
        void setReserveBlocksForTest(int64_t reserve) {
            reserveBlocks_ = reserve;
        }

        /// @brief Override the eviction size floor (`kMinEvictBlocks`) for a
        ///        unit test; production uses the named constant.
        ///
        /// @details
        /// The drain's tier-1 scan skips LBs holding fewer than the floor's
        /// blocks (tier 2 of `pickVictimTwoTier` is always floorless). A unit
        /// test that pins small fixture LBs (to stay fast) lowers the floor so
        /// they qualify in tier 1; the floor's own behaviour is covered by the
        /// direct `pickVictimBehindCursor` and `pickVictimTwoTier` tests. A
        /// pure test seam — production never calls it, never a proof input.
        ///
        /// @param minBlocks The size floor in blocks for subsequent passes.
        void setMinEvictBlocksForTest(int64_t minBlocks) {
            minEvictBlocks_ = minBlocks;
        }

        /// @brief The unified per-worker claim+load handshake for all three
        ///        phases (the user's "a worker uploads itself").
        ///
        /// @details
        /// Claims `lb` for the calling worker through the claim word and makes
        /// it resident before the worker reads it. An LB already `WorkerOwned`
        /// (a split sibling) returns immediately; `Idle`/`Dumped` CAS to
        /// `WorkerOwned`, and a cold LB is reloaded — normally straight into
        /// the free blocks the reserve-target drain holds, with NO eviction.
        /// The load↔evict exchange is gone; the WIDENED WORKER PRESSURE VALVE
        /// replaces it: EVERY claim (resident or cold — the pure-grant path,
        /// where a worker's task grants scratch / statement blocks with no
        /// reload in sight, needs the valve just as much) checks the pool's
        /// free blocks against the emergency floor
        /// (`kEmergencyFloorBlocks` = reserve/4) and, when breached, evicts
        /// ONE behind-cursor victim (`evictOneForReload`, two-tier Belady) and
        /// counts it (`emergencyEvictCount`, expected 0 in a healthy reserve
        /// regime — a nonzero count is the alarm the planner cannot keep up).
        /// The cold-reload miss path adds a second trigger: free blocks that
        /// cannot cover THIS reload (estimated from the LB's last raw-image
        /// byte count) also fire the valve. At most ONE eviction per claim,
        /// no loop: if one eviction is not enough the reload's `acquireBlock`
        /// surfaces genuine exhaustion (Rule 19), never a silent wait. `Busy`
        /// means the steward is mid-op on this exact LB; the worker
        /// yield-spins, bounded by `kStuckSeconds` — a longer wait asserts
        /// ([I-113], the forward-progress backstop). The caller releases the
        /// claim (`stewardClaim = Idle`) when it finishes the LB, making it
        /// deloadable again.
        ///
        /// `phase` buckets this handshake's telemetry (claim-wait, self-load,
        /// inline-evict, prefetch hit/miss) — pure telemetry, never steers the
        /// handshake.
        ///
        /// @param lb        The LB the worker is about to process.
        /// @param phase     The phase id, 1..4 (4 = barrier seam) — the telemetry bucket.
        /// @param directory The deload directory for reloads / evictions.
        void claimAndLoadForWork(Memory& lb, int phase,
                                 const std::string& directory);

        /// @brief One PLANNER pass over an open phase window: enqueue the
        ///        prefetch loads, the reshuffles, and the reserve-shortfall
        ///        evictions onto the I/O rings — the executors do the I/O.
        ///
        /// @details
        /// Runs on the planner thread with the mutex released; parameters are
        /// the snapshot taken under the mutex. The planner does NO I/O itself
        /// (the discharge drain, which stays on the planner thread, is the
        /// one exception — see `threadMain`). Two parts. (1) PREFETCH
        /// planning: over the UNIFIED prefetch region `[cursor, cursor +
        /// workers + kLookaheadWorkerMultiple * workers)` — starting AT the
        /// cursor and ending where the keep window ends — enqueue a HIGH-lane
        /// load task for each `Dumped` LB and a LOW-lane reshuffle task for
        /// each fragmented `Idle` one (`needsReshuffle`); load enqueues stop
        /// at the PREFETCH BUDGET (pending HIGH-lane load blocks summed
        /// under the mutex, symmetric to the eviction projection, plus this
        /// pass's issues, capped at `reserveBlocks_ / kPrefetchBudgetDivisor`
        /// — pending loads always fit inside the reserve's free blocks).
        /// (2) DRAIN planning:
        /// while the PROJECTED free blocks — live free + in-flight dump
        /// blocks (`inFlightBlocks`, freed-but-unavailable) + blocks of
        /// eviction tasks already enqueued — fall short of `kReserveBlocks`,
        /// walk victims backward from `keepLo - 1` in Belady order (two-tier:
        /// the floor-preferred lap first, a floorless lap when the reserve is
        /// still short) and enqueue a LOW-lane eviction task per victim.
        /// The pass ends when the projection meets the reserve, no victim
        /// remains (a DEFINED result — the working set fits, the stream
        /// pauses; genuine exhaustion still asserts at `acquireBlock`,
        /// Rule 19), or the ring fills (the designed load-shedding — counted,
        /// re-planned next pass). Duplicate enqueues are suppressed by a ring
        /// containment check (hygiene; the claim-word CAS at execution is the
        /// correctness arbiter). The working set kept resident is `[cursor -
        /// workers, cursor + workers + kLookaheadWorkerMultiple * workers)`:
        /// a freshly prefetched LB is INSIDE the kept window and never
        /// planned for eviction by the same pass. Every enqueued task
        /// arbitrates through the claim word at execution and is
        /// content-invisible, so pass timing never shifts an observable
        /// ([I-103]). When the projection meets `kReserveBlocks` nothing is
        /// planned, so a batch under the reserve deficit sees zero
        /// deload/reload traffic. Public so a unit test can drive one
        /// synchronous planning pass (paired with `drainIoRingsForTest`)
        /// under controlled pressure — the planner thread calls it
        /// identically.
        ///
        /// @param cursor  The current phase's dispatch cursor.
        /// @param order   The kernel's active vector.
        /// @param workers The in-flight margin (worker count). (The deload
        ///                directory is no longer a parameter — the planner
        ///                does no I/O; executing tasks read the window's
        ///                registered directory.)
        void maintainWorkingSet(const std::atomic<std::size_t>* cursor,
                                const std::vector<Memory*>* order,
                                unsigned workers);

        /// @brief Blocks currently held by in-flight eviction dumps —
        ///        freed-but-unavailable in the reserve accounting.
        ///
        /// @details
        /// An eviction task holds its LB's blocks from the claim until the
        /// raw image is fully written and the blocks returned (the async dump
        /// contract: `Dumped` is never observable before both). During that
        /// window the blocks are neither free nor part of any working set;
        /// the planner's reserve projection counts them as already-freed so
        /// it never enqueues a second wave of evictions to cover the same
        /// shortfall. Relaxed atomic — telemetry-grade accounting, never a
        /// proof input.
        ///
        /// @return Blocks held by eviction dumps currently executing.
        int64_t inFlightBlocks() const {
            return inFlightDumpBlocks_.load(std::memory_order_relaxed);
        }

        /// @brief Bytes held by in-flight eviction dumps (the block-granular
        ///        twin of `inFlightBlocks`).
        ///
        /// @return `inFlightBlocks()`-worth of bytes at the pool's block
        ///         size, accumulated per task at its claim.
        int64_t inFlightBytes() const {
            return inFlightDumpBytes_.load(std::memory_order_relaxed);
        }

        /// @brief Number of tasks currently queued on one I/O ring — a unit
        ///        test's dedup / drop observation point.
        ///
        /// @details A pure test seam (production never reads ring depths);
        ///          takes the mutex, so it is safe against a live planner.
        ///
        /// @param high `true` = the HIGH (prefetch-load) ring, `false` = the
        ///             LOW (eviction + reshuffle) ring.
        /// @return The ring's current task count.
        std::size_t ioRingDepthForTest(bool high) const;

        /// @brief Pop and execute exactly ONE I/O task on the calling thread,
        ///        honoring the executor lane-priority rule.
        ///
        /// @details
        /// A pure test seam that makes the lane-priority flip directly
        /// observable: the pop applies the same rule the executors use (HIGH
        /// first, EXCEPT LOW first when free blocks sit below the emergency
        /// floor). Executes with the steward's own scratch arena. Intended
        /// for a steward whose executor pool is not running (never started,
        /// or quiesced) — the seam does not coordinate with live executors
        /// beyond the mutex.
        ///
        /// @return 0 = both rings empty (nothing ran), 1 = a HIGH task ran,
        ///         2 = a LOW task ran.
        int runOneIoTaskForTest();

        /// @brief Execute every queued I/O task on the calling thread (pop in
        ///        executor priority order until both rings are empty).
        ///
        /// @details
        /// A pure test seam: lets a test drive `maintainWorkingSet` planning
        /// synchronously and then apply the plan deterministically without a
        /// live executor pool. Semantically one full drain of the rings by a
        /// single executor.
        void drainIoRingsForTest();

        /// @brief Redirect the pressure reads (`totalBlocks` / `blocksInUse`)
        ///        to a test-supplied pool; production leaves this null and
        ///        reads the process `staticMemory()`.
        ///
        /// @details
        /// The pager's drain decisions read the pool's block ledger; the
        /// process pool is 4 GiB, so a unit test cannot reach a reserve deficit
        /// by acquire-and-hold. A test binds a small private
        /// `GlobalMemoryManager` here, acquires/holds its blocks to force the
        /// pressure it wants to exercise, then clears it (pass `nullptr`).
        /// A pure test seam — NOT a production code path (production never
        /// calls it), never a proof input.
        ///
        /// @param pool The pool to read pressure from, or `nullptr` to restore
        ///             the default (`staticMemory()`).
        void setPressurePoolForTest(const GlobalMemoryManager* pool) {
            pressurePool_ = pool;
        }

    private:
        /// @brief The pool the pager reads pressure from — the test override
        ///        `pressurePool_` when set, else the process `staticMemory()`.
        ///
        /// @details
        /// Every `totalBlocks()` / `blocksInUse()` read in the drain and the
        /// worker emergency gate routes through here so a test can inject a
        /// small pool. The alternative branches are both defined results (an
        /// injected pool or the default) — not a "couldn't, so degrade"
        /// fallback (Rule 19).
        ///
        /// @return The pool to read block-pressure counters from.
        const GlobalMemoryManager& pressurePool() const {
            return pressurePool_ ? *pressurePool_ : staticMemory();
        }

        /// @brief One queued I/O unit: a prefetch load, an eviction dump, or
        ///        a reshuffle, bound to the phase window that planned it.
        ///
        /// @details
        /// `lb` is the address-stable target (`Memory` objects never move,
        /// I-109); `idx` is its position in the planning window's order
        /// vector, used ONLY for the eviction/reshuffle re-validation against
        /// the CURRENT cursor (never dereferenced into a possibly-dead
        /// vector); `windowGen` binds the task to the window generation that
        /// planned it — a task whose window closed drops at execution
        /// (`kAnyWindowGeneration` opts out, for the barrier head prefetch).
        /// Plain POD so the fixed rings stay heap-free.
        struct IoTask {
            enum class Kind : uint8_t { Load, Evict, Reshuffle };
            Kind kind;
            Memory* lb;
            std::size_t idx;
            uint64_t windowGen;
        };

        /// @brief Outcome of one ring enqueue — all three are defined
        ///        results (Rule 19), and the planner branches on them:
        ///        `Queued` counts the victim's blocks into the projection,
        ///        `Duplicate` does not (the queued twin already did),
        ///        `Dropped` ends the planning pass (load-shedding — the
        ///        whole plan regenerates next pass).
        enum class IoEnqueue : uint8_t { Queued, Duplicate, Dropped };

        /// @brief Thread body of the PLANNER: park on the condition variable;
        ///        run a planning pass per poll tick while a window is open;
        ///        drain the installed discharge list on wake; repeat until
        ///        stop.
        ///
        /// @details The discharge drain is the planner's ONE remaining I/O
        ///          duty (I-106 execution unchanged); all window I/O runs on
        ///          the executor pool.
        void threadMain();

        /// @brief Thread body of one I/O EXECUTOR: pop tasks in lane-priority
        ///        order and execute them until stop.
        ///
        /// @details
        /// Lane priority: HIGH (prefetch loads) first, EXCEPT when free
        /// blocks sit below the emergency floor — then LOW (evictions +
        /// reshuffles) first, so evictions cannot starve behind a reload
        /// burst that has no blocks to land in. Task execution is
        /// claim-first drop-on-lose; every task is idempotent and
        /// stale-tolerant, so executor scheduling is content-invisible.
        ///
        /// @param execIndex This executor's index (selects its private
        ///                  scratch arena).
        void ioThreadMain(unsigned execIndex);

        /// @brief Enqueue one task onto a ring (mutex taken inside): dedup by
        ///        (kind, lb) containment, drop-and-count when full, notify.
        ///
        /// @details
        /// The containment dedup is hygiene against the 200 µs planner
        /// re-enqueueing the same not-yet-executed task every pass (the
        /// claim-word CAS at execution is the correctness arbiter either
        /// way). A full ring DROPS the enqueue — the designed load-shedding
        /// (the planner regenerates next pass), counted per lane
        /// (`prefetchDroppedFullRing` / `evictDroppedFullRing`). Both
        /// outcomes are defined results (Rule 19).
        ///
        /// @param task The task to queue.
        /// @param high `true` = HIGH ring (loads), `false` = LOW ring
        ///             (evictions + reshuffles).
        /// @return The enqueue outcome (see `IoEnqueue`).
        IoEnqueue enqueueIoTask(const IoTask& task, bool high);

        /// @brief Pop the next task in lane-priority order. Caller holds
        ///        `mutex_`.
        ///
        /// @details
        /// Applies the executor priority rule (HIGH first; LOW first below
        /// the emergency floor). The pressure read happens through
        /// `pressurePool()` while holding the steward mutex — the lock order
        /// steward-mutex → pool-mutex is used everywhere, never the reverse,
        /// so no inversion is possible.
        ///
        /// @param out Receives the popped task on success.
        /// @return `true` with `out` filled, `false` when both rings empty.
        bool popIoTaskLocked(IoTask& out);

        /// @brief Execute one popped task: re-validate, claim, do the I/O,
        ///        release.
        ///
        /// @details
        /// Load: drop when the task's window generation is stale (unless
        /// `kAnyWindowGeneration`); CAS `Dumped → Busy` (drop on lose);
        /// `ensureLoaded`; release `Idle`. Evict: re-validate the window
        /// under the CURRENT cursor (drop when the window closed, the
        /// generation moved, or the index slid inside the kept range);
        /// eligibility pre-check; CAS `Idle → Busy`; account the blocks
        /// in-flight; raw dump + release (`deloadStaticContainersRaw` — the
        /// image is complete on disk and the blocks returned BEFORE the
        /// `Dumped` store, the async dump contract); store `Dumped`.
        /// Reshuffle: same re-validation; re-check `needsReshuffle`; CAS
        /// `Idle → Busy`; compact on `scratch`; release `Idle`. Every drop is
        /// a defined hand-over (a worker or a newer plan owns the LB), never
        /// a failure.
        ///
        /// @param task    The task to execute.
        /// @param scratch The executing thread's private compaction scratch.
        void executeIoTask(const IoTask& task, LbArena& scratch);

        mutable std::mutex mutex_;
        std::condition_variable cv_;
        std::thread thread_;
        std::vector<Memory*> work_;
        std::vector<Memory*> evictionWork_;
        std::string directory_;
        // Scratch arena for the test-seam ring drain (runOneIoTaskForTest /
        // drainIoRingsForTest execute reshuffles on the calling thread): drawn
        // from the never-deloaded LB-body pool, NOT the deloadable pool it
        // compacts. The executor pool owns one private arena per executor
        // (ioScratch_); the discharge path has its own.
        LbArena stewardScratch_{ &lbMemory() };
        const std::atomic<std::size_t>* windowCursor_ = nullptr;
        const std::vector<Memory*>* windowOrder_ = nullptr;
        unsigned windowWorkers_ = 0;
        // Monotone window identity: bumped at every beginPhaseWindow AND
        // endPhaseWindow, so a task can tell whether the window that planned
        // it is still the open one (an address compare on order vectors could
        // alias across pass-local vectors; the generation cannot).
        uint64_t windowGeneration_ = 0;
        // Pressure-read override for unit tests only (null in production ->
        // staticMemory()); see pressurePool() / setPressurePoolForTest.
        const GlobalMemoryManager* pressurePool_ = nullptr;
        // The reserve-target the drain maintains; production = kReserveBlocks,
        // overridable by a unit test (setReserveBlocksForTest).
        int64_t reserveBlocks_ = steward::kReserveBlocks;
        // The eviction size floor; production = kMinEvictBlocks, overridable by
        // a unit test (setMinEvictBlocksForTest).
        int64_t minEvictBlocks_ = steward::kMinEvictBlocks;
        // The two fixed-capacity I/O task rings (heap-free after construction;
        // guarded by mutex_). HIGH = prefetch loads; LOW = evictions +
        // reshuffles.
        IoTask highRing_[steward::kIoRingCapacity];
        std::size_t highHead_ = 0;
        std::size_t highCount_ = 0;
        IoTask lowRing_[steward::kIoRingCapacity];
        std::size_t lowHead_ = 0;
        std::size_t lowCount_ = 0;
        // The I/O executor pool: threads + one private compaction scratch
        // arena per executor (deque: LbArena is not movable; entries are
        // created once per start()).
        std::vector<std::thread> ioThreads_;
        std::deque<LbArena> ioScratch_;
        // Executors currently executing a task (guarded by mutex_); part of
        // the quiesce condition.
        unsigned ioBusy_ = 0;
        // Freed-but-unavailable accounting for in-flight eviction dumps (the
        // async dump contract); see inFlightBlocks() / inFlightBytes().
        std::atomic<int64_t> inFlightDumpBlocks_{ 0 };
        std::atomic<int64_t> inFlightDumpBytes_{ 0 };
        bool runRequested_ = false;
        bool busy_ = false;
        bool stopRequested_ = false;
        bool drainedSinceInstall_ = false;
        bool running_ = false;
    };

    /// @brief Print the one-shot `[EXHAUSTION]` block census to stderr — the
    ///        forensic body of the pool-exhaustion reporter.
    ///
    /// @details
    /// Resolves the survival fork at the wall: POLICY-MISSED evictable
    /// blocks (a large "eligible victims" line — the pager left reclaimable
    /// blocks on the table) versus a GENUINELY PINNED working set (large
    /// `WorkerOwned` + scratch + `UNATTRIBUTED` lines — the concurrent
    /// demand alone exceeds the pool). Census lines, all `[EXHAUSTION]`-
    /// prefixed for grep: per-claim-state LB counts + summed `blocksHeld`;
    /// the Idle+resident ELIGIBLE-VICTIM bucket (active, unsplit,
    /// `blocksHeld >= 1`); the steward's in-flight dump blocks/bytes; the
    /// two scratch registries' per-slot block sums (string tier + request-
    /// generation tier); the reserve / emergency-floor constants; the
    /// cumulative `DeloadStats` counters relevant at the wall (emergency
    /// evictions, ring drops, prefetch issued/hit/miss, pager evictions);
    /// and `UNATTRIBUTED` = `blocksInUse` − grid `blocksHeld` − in-flight −
    /// scratch — the residue that covers sealed page sets, CE-clone LB
    /// arenas (not reachable from any registry — stated in the header
    /// line), and racy skew.
    ///
    /// MUTEX CONTRACT: runs under the POOL MUTEX (the exhaustion-reporter
    /// contract, `GlobalMemoryManager::setExhaustionReporter`). Every read
    /// is manager-mutex-free by construction: claim words and telemetry are
    /// atomics; `blocksHeld()` / `resident()` / `isActive` /
    /// `numberOfParts` are plain per-object bookkeeping; the pool counters
    /// arrive as parameters. Reads of other threads' live arenas are racy —
    /// the process is about to abort, the census is approximate by design
    /// (never a proof input; pure post-mortem observability, Rule 16).
    ///
    /// @param grid        The prover's LB grid (`prove()`'s `bodies`).
    /// @param steward     The live steward (its in-flight dump accounting).
    /// @param blocksInUse Pool blocks granted at the wall (passed in by the
    ///                    manager — the reporter must not fetch it).
    /// @param totalBlocks Pool capacity in blocks (passed in likewise).
    void printExhaustionCensus(const std::vector<Memory*>& grid,
                               const MemorySteward* steward,
                               int64_t blocksInUse,
                               int64_t totalBlocks);
}
