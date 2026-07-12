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
#include <chrono>
#include <cstdint>
#include <vector>

namespace gl {

    /// @brief The per-phase deload traffic captured over one working-set
    ///        window (`DeloadStats::beginPhaseWindow` →
    ///        `DeloadStats::endPhaseWindow`).
    ///
    /// @details
    /// Pure telemetry payload: the difference between the process-wide
    /// counters at the window's close and their snapshot at its open, plus
    /// the window's own wall duration. Returned by `endPhaseWindow` so a
    /// caller (or a unit test) can read the deltas the printed line reports;
    /// it never feeds a prover decision (Rule 16 / [I-44]). `phase` is the
    /// 1/2/3 phase id the window was opened for.
    struct PhaseDelta {
        int phase = 0;              // phase id (1..4; 4 = barrier seam)
        int64_t outBytes = 0;       // dump bytes streamed (v3 + v4-raw)
        int64_t outNs = 0;          // dump time (v3 + v4-raw)
        int64_t inBytes = 0;        // load bytes streamed (v3 + v4-raw)
        int64_t inNs = 0;           // load time (v3 + v4-raw)
        int64_t workerWaitNs = 0;   // worker Busy-claim wait time, this phase
        int64_t selfLoads = 0;      // worker self-loads during the window
        int64_t inlineEvicts = 0;   // worker inline evictions during the window
        double wallSeconds = 0.0;   // the window's own wall-clock duration
    };

    /// @brief Process-wide deload/reload telemetry aggregate — the single
    ///        merged datapath + steward counter package.
    ///
    /// @details
    /// One instance per process (`deloadStats()`), holding every deload,
    /// reload, index-rebuild, eviction, prefetch, and worker-stall counter
    /// as `std::atomic<int64_t>` written with relaxed ordering (a monotone
    /// counter needs no synchronization — the reader is a single-threaded
    /// end-of-batch report). Timers use `std::chrono::steady_clock`.
    ///
    /// Nothing in the prover ever READS these counters to steer control flow
    /// — they are pure telemetry (Rule 16 / [I-44]): the deload set, the
    /// eviction victims, and the theorem output are all independent of every
    /// value here. The aggregate is reset per batch (`reset()`), printed once
    /// per batch (`reportSummary()` beside the `[STATIC-MEMORY]` line and, at
    /// each phase window's close, an optional per-phase `[DELOAD]` line), and
    /// carries a once-per-batch LB-size histogram guard.
    ///
    /// @invariant No counter is ever read by a prover decision — only by the
    ///            reports and by unit tests.
    /// @see `deloadStats()`, `PhaseDelta`, `reportSummary`, `reportLbHistogram`.
    struct DeloadStats {
        // ---- v3 canonical element-stream dump / load (bytes + ns + count) ---
        std::atomic<int64_t> v3DumpBytes{ 0 };   // payload bytes streamed out
        std::atomic<int64_t> v3DumpNs{ 0 };      // dump streaming time
        std::atomic<int64_t> v3DumpCount{ 0 };   // dump operations (base + tail)
        std::atomic<int64_t> v3LoadBytes{ 0 };   // payload bytes streamed in
        std::atomic<int64_t> v3LoadNs{ 0 };      // load streaming time
        std::atomic<int64_t> v3LoadCount{ 0 };   // load operations

        // ---- v4 raw arena-image dump / load (bytes + ns + count) -----------
        // The near-memcpy eviction/reload datapath: page bytes stream straight
        // from / into pool memory, no element walk, no per-key index rebuild.
        std::atomic<int64_t> rawDumpBytes{ 0 };  // raw page bytes streamed out
        std::atomic<int64_t> rawDumpNs{ 0 };     // raw dump streaming time
        std::atomic<int64_t> rawDumpCount{ 0 };  // raw dump operations
        std::atomic<int64_t> rawLoadBytes{ 0 };  // raw page bytes streamed in
        std::atomic<int64_t> rawLoadNs{ 0 };     // raw load streaming time
        std::atomic<int64_t> rawLoadCount{ 0 };  // raw load operations
        std::atomic<int64_t> rawLoadPeakNs{ 0 }; // slowest single raw reload —
                                                 // the reload cost model's tail

        // ---- reload-driven throw-away index rebuild ------------------------
        std::atomic<int64_t> indexRebuildNs{ 0 };    // rebuild time on reload
        std::atomic<int64_t> indexRebuildKeys{ 0 };  // keys re-placed on reload
        std::atomic<int64_t> indexRebuildCount{ 0 };  // rebuild CALLS (a raw
                                                     // reload must add zero)

        // ---- dumpStaticContainers hygiene (the three-way deload decision) --
        std::atomic<int64_t> evictSkippedClean{ 0 };  // Clean + base = no write
        std::atomic<int64_t> tailDumpCount{ 0 };      // tail-delta dumps
        std::atomic<int64_t> fullDumpCount{ 0 };      // full canonical dumps

        // ---- steward working-set-pager evictions (maintainWorkingSet) ------
        std::atomic<int64_t> evictCount{ 0 };    // LBs evicted by the pager
        std::atomic<int64_t> evictBlocks{ 0 };   // blocks freed by those evictions
        std::atomic<int64_t> evictDumpNs{ 0 };   // eviction dump+release time

        // ---- prefetch ------------------------------------------------------
        std::atomic<int64_t> prefetchIssued{ 0 };  // steward prefetch reloads
        std::atomic<int64_t> prefetchHit{ 0 };     // worker found the LB ready
        std::atomic<int64_t> prefetchMiss{ 0 };    // worker had to self-load

        // ---- I/O ring load-shedding (planner enqueues onto a full ring) ----
        // A drop is the DESIGNED overflow behavior, not a failure: the planner
        // regenerates its whole plan on the next poll pass, so a dropped task
        // is merely re-planned. The counters make sustained shedding visible
        // (a persistently full LOW ring = the executors cannot keep up with
        // the eviction demand).
        std::atomic<int64_t> prefetchDroppedFullRing{ 0 };  // HIGH-ring drops
        std::atomic<int64_t> evictDroppedFullRing{ 0 };     // LOW-ring drops

        // ---- worker self-load + inline eviction (claimAndLoadForWork) ------
        std::atomic<int64_t> workerSelfLoadNs{ 0 };     // worker ensureLoaded time
        std::atomic<int64_t> workerSelfLoadCount{ 0 };  // worker self-loads
        std::atomic<int64_t> workerInlineEvictNs{ 0 };     // inline evict time
        std::atomic<int64_t> workerInlineEvictCount{ 0 };  // inline evictions
        // The counted EMERGENCY worker eviction (claimAndLoadForWork's fallback
        // when the pool is above the 3/4 hard bound AND the free blocks cannot
        // cover this reload). Expected 0 in a healthy reserve regime; a nonzero
        // count is the ALARM that the reserve target is undersized.
        std::atomic<int64_t> emergencyEvictCount{ 0 };     // emergency evictions

        // ---- worker Busy-claim wait, per phase (index 1..4; 0 unused; 4 =
        // the barrier seam window between iterations) ----------------------
        std::atomic<int64_t> workerClaimWaitNs[5]{};
        std::atomic<int64_t> workerClaimWaitCount[5]{};

        /// @brief Record one v3 canonical dump (base or tail file set).
        ///
        /// @details
        /// A single relaxed accumulation of the dump's streamed payload byte
        /// count, its streaming time, and the operation count. Called from the
        /// serializer (`lbdeload::dumpLbMemory` / `dumpLbMemoryTail`), where
        /// the operation is already slow enough that the timer read is free.
        ///
        /// @param bytes Payload bytes the dump streamed; >= 0.
        /// @param ns    Wall time the dump took, nanoseconds; >= 0.
        void recordV3Dump(int64_t bytes, int64_t ns) {
            v3DumpBytes.fetch_add(bytes, std::memory_order_relaxed);
            v3DumpNs.fetch_add(ns, std::memory_order_relaxed);
            v3DumpCount.fetch_add(1, std::memory_order_relaxed);
        }

        /// @brief Record one v3 canonical reload (a full file-set replay).
        ///
        /// @details
        /// The read-path twin of `recordV3Dump`: one relaxed accumulation of
        /// the reload's payload byte count, its time, and the operation count.
        /// Called from `lbdeload::loadLbMemory`.
        ///
        /// @param bytes Payload bytes the reload streamed; >= 0.
        /// @param ns    Wall time the reload took, nanoseconds; >= 0.
        void recordV3Load(int64_t bytes, int64_t ns) {
            v3LoadBytes.fetch_add(bytes, std::memory_order_relaxed);
            v3LoadNs.fetch_add(ns, std::memory_order_relaxed);
            v3LoadCount.fetch_add(1, std::memory_order_relaxed);
        }

        /// @brief Record one v4 raw arena-image dump.
        ///
        /// @details
        /// The near-memcpy write path's twin of `recordV3Dump`: one relaxed
        /// accumulation of the raw image's streamed page byte count, its
        /// streaming time, and the operation count. Called from
        /// `lbdeload::dumpLbMemoryRaw`, where the file write is already the
        /// slow part so the timer read is free.
        ///
        /// @param bytes Page bytes the raw dump streamed; >= 0.
        /// @param ns    Wall time the raw dump took, nanoseconds; >= 0.
        void recordRawDump(int64_t bytes, int64_t ns) {
            rawDumpBytes.fetch_add(bytes, std::memory_order_relaxed);
            rawDumpNs.fetch_add(ns, std::memory_order_relaxed);
            rawDumpCount.fetch_add(1, std::memory_order_relaxed);
        }

        /// @brief Record one v4 raw arena-image reload.
        ///
        /// @details
        /// The read-path twin of `recordRawDump`: one relaxed accumulation of
        /// the raw reload's page byte count, its time, and the operation count.
        /// Called from `lbdeload::loadLbMemoryRaw`. A raw reload rebuilds no
        /// throw-away index, so it must leave `indexRebuildCount` untouched —
        /// the property the C2 tests assert against.
        ///
        /// @param bytes Page bytes the raw reload streamed; >= 0.
        /// @param ns    Wall time the raw reload took, nanoseconds; >= 0.
        void recordRawLoad(int64_t bytes, int64_t ns) {
            rawLoadBytes.fetch_add(bytes, std::memory_order_relaxed);
            rawLoadNs.fetch_add(ns, std::memory_order_relaxed);
            rawLoadCount.fetch_add(1, std::memory_order_relaxed);
            // Track the slowest single reload (the cost model's worst case). A
            // relaxed CAS loop: contention is rare (reloads are already the slow
            // path) and the value is telemetry, never a prover input.
            int64_t prev = rawLoadPeakNs.load(std::memory_order_relaxed);
            while (ns > prev
                && !rawLoadPeakNs.compare_exchange_weak(
                       prev, ns, std::memory_order_relaxed)) {
                // prev is reloaded by compare_exchange_weak on failure; retry.
            }
        }

        /// @brief Record one reload-driven throw-away index rebuild.
        ///
        /// @details
        /// Times the `rebuildIndex()` a cold-map key reload triggers and
        /// counts the keys re-placed AND the rebuild call. Called only on the
        /// v3 reload path (the `KeysView::bulkAppendBytes` → `rebuildIndex`
        /// hook), never on the hot mint / growth path, so the timer read never
        /// lands on a fast path. The v4 raw reload path never calls this — the
        /// raw image carries the bucket pages verbatim — so `indexRebuildCount`
        /// staying 0 across a raw reload is the proof the index was preserved.
        ///
        /// @param keys Keys re-placed into the fresh index; >= 0.
        /// @param ns   Wall time the rebuild took, nanoseconds; >= 0.
        void recordIndexRebuild(int64_t keys, int64_t ns) {
            indexRebuildKeys.fetch_add(keys, std::memory_order_relaxed);
            indexRebuildNs.fetch_add(ns, std::memory_order_relaxed);
            indexRebuildCount.fetch_add(1, std::memory_order_relaxed);
        }

        /// @brief Record one steward working-set-pager eviction.
        ///
        /// @details
        /// One relaxed accumulation of the eviction count, the blocks the
        /// eviction returned to the pool, and the dump+release time. Called
        /// from the executor-pool eviction task
        /// (`MemorySteward::executeIoTask`, planned by `maintainWorkingSet`;
        /// not the worker inline path, which is `recordWorkerInlineEvict`).
        ///
        /// @param blocks Blocks the eviction freed (`blocksHeld` before the
        ///               release); >= 0.
        /// @param ns     Wall time the deload+release took, nanoseconds; >= 0.
        void recordEviction(int64_t blocks, int64_t ns) {
            evictCount.fetch_add(1, std::memory_order_relaxed);
            evictBlocks.fetch_add(blocks, std::memory_order_relaxed);
            evictDumpNs.fetch_add(ns, std::memory_order_relaxed);
        }

        /// @brief Record one worker self-load (a prefetch miss's reload).
        ///
        /// @details
        /// Accumulates the `ensureLoaded` time and the self-load count when a
        /// worker in `claimAndLoadForWork` finds its LB cold and reloads it
        /// itself (the prefetch did not land it in time — `prefetchMiss` is
        /// bumped separately at the same site).
        ///
        /// @param ns Wall time the worker's reload took, nanoseconds; >= 0.
        void recordWorkerSelfLoad(int64_t ns) {
            workerSelfLoadNs.fetch_add(ns, std::memory_order_relaxed);
            workerSelfLoadCount.fetch_add(1, std::memory_order_relaxed);
        }

        /// @brief Record one worker inline eviction (make-room-to-reload).
        ///
        /// @details
        /// Accumulates the `evictOneForReload` time and count when a
        /// self-loading worker had to free room before its reload. Distinct
        /// from `recordEviction` (the steward's pager evictions).
        ///
        /// @param ns Wall time the inline eviction took, nanoseconds; >= 0.
        void recordWorkerInlineEvict(int64_t ns) {
            workerInlineEvictNs.fetch_add(ns, std::memory_order_relaxed);
            workerInlineEvictCount.fetch_add(1, std::memory_order_relaxed);
        }

        /// @brief Record one worker Busy-claim wait, bucketed by phase.
        ///
        /// @details
        /// Accumulates the time a worker spent spinning on a `Busy` claim
        /// (the steward or a split sibling mid-op on the same LB) and the wait
        /// count, into the phase's slot. Timed only once a wait actually began
        /// — the resident fast path records nothing here.
        ///
        /// @param phase The phase id, 1..4 (asserted in range; 4 = the
        ///              barrier seam window).
        /// @param ns    Wall time spent waiting, nanoseconds; >= 0.
        void recordWorkerClaimWait(int phase, int64_t ns);

        /// @brief Open the per-phase telemetry window: snapshot the counters
        ///        this phase's `[DELOAD]` line will diff, and stamp the wall
        ///        start.
        ///
        /// @details
        /// Kernel-thread only (single-threaded between phases): no window may
        /// already be open (asserted). Records the phase id, a steady-clock
        /// start, and a relaxed snapshot of the dump/load byte+time counters,
        /// the worker self-load / inline-evict counts, and this phase's
        /// claim-wait time. `endPhaseWindow` diffs against this snapshot.
        ///
        /// @param phase The phase id, 1..4 (asserted in range; 4 = the
        ///              barrier seam window between iterations).
        void beginPhaseWindow(int phase);

        /// @brief Close the per-phase telemetry window: compute the deltas,
        ///        print the `[DELOAD]` line when the phase moved any deload
        ///        traffic, and return the deltas.
        ///
        /// @details
        /// Kernel-thread only, after the phase join (asserts a window is
        /// open). Diffs the current counters against the `beginPhaseWindow`
        /// snapshot, measures the window's own wall duration, and — only when
        /// the phase saw nonzero out/in bytes, worker wait, self-loads, or
        /// inline evicts — prints one line to stdout. Always returns the
        /// computed `PhaseDelta` (for tests / callers); the caller may ignore
        /// it.
        ///
        /// @return The window's deload deltas and wall duration.
        PhaseDelta endPhaseWindow();

        /// @brief Print the end-of-batch `[DELOAD]` summary block to stdout.
        ///
        /// @details
        /// One multi-line block reporting every accumulated total (dump /
        /// load volumes and rates, index rebuild, evictions and hygiene,
        /// prefetch, worker self-loads / inline evicts, and per-phase
        /// claim-wait). Single-threaded call at batch end, right after the
        /// `[STATIC-MEMORY]` line; the caller then `reset()`s for the next
        /// batch. Pure output — reads counters, changes no state.
        void reportSummary() const;

        /// @brief Has this batch's one-shot LB-size histogram not yet printed?
        ///
        /// @details
        /// A cheap guard the kernel checks before assembling the histogram
        /// input, so the (once-per-batch) `blocksHeld` gather runs only on the
        /// batch's first kernel barrier. `reportLbHistogram` sets the flag.
        ///
        /// @return `true` until `reportLbHistogram` has run this batch.
        bool lbHistogramPending() const {
            return !lbHistogramPrinted_.load(std::memory_order_relaxed);
        }

        /// @brief Print the once-per-batch active-LB / `blocksHeld` histogram
        ///        and mark it printed.
        ///
        /// @details
        /// Reports the active-LB count and the `blocksHeld` distribution
        /// (max / p95 / median / count-below-`kSmallLbBlocks`) — the raw
        /// inputs for later tuning constants (the eviction size floor, the
        /// reserve target). Sorts its input in place. Sets the once-per-batch
        /// guard so a second call this batch is a defined no-op.
        ///
        /// @param activeCount   Active LBs at this barrier; >= 0.
        /// @param blocksHeld    Per-LB block counts (resident LBs), any order;
        ///                      sorted in place by this call.
        void reportLbHistogram(int activeCount, std::vector<int64_t>& blocksHeld);

        /// @brief Reset every counter and the histogram guard for a new batch.
        ///
        /// @details
        /// Single-threaded batch-boundary call (steward stopped). Stores 0 to
        /// every atomic and clears the once-per-batch histogram flag; leaves
        /// no phase window open (asserts none is).
        void reset();

        /// @brief Blocks-held threshold below which an LB is a "small" LB in
        ///        the histogram's count-below bucket (1 MiB at 256 KiB blocks).
        static constexpr int64_t kSmallLbBlocks = 4;

    private:
        // ---- once-per-batch histogram guard -------------------------------
        std::atomic<bool> lbHistogramPrinted_{ false };

        // ---- open phase window (kernel-thread only, single-threaded) ------
        int phaseWindowId_ = 0;   // 0 = closed; 1..4 = open for that phase
        std::chrono::steady_clock::time_point phaseWindowStart_{};
        int64_t snapDumpBytes_ = 0;
        int64_t snapDumpNs_ = 0;
        int64_t snapLoadBytes_ = 0;
        int64_t snapLoadNs_ = 0;
        int64_t snapRawDumpBytes_ = 0;   // v4-raw dump, folded into out totals
        int64_t snapRawDumpNs_ = 0;
        int64_t snapRawLoadBytes_ = 0;   // v4-raw load, folded into in totals
        int64_t snapRawLoadNs_ = 0;
        int64_t snapSelfLoadCount_ = 0;
        int64_t snapInlineEvictCount_ = 0;
        int64_t snapClaimWaitNs_ = 0;
    };

    /// @brief The process-wide deload telemetry aggregate.
    ///
    /// @details
    /// A function-local static — one `DeloadStats` for the whole process,
    /// constructed on first use. Every dump / load / eviction / prefetch site
    /// records into it and the end-of-batch report reads it; nothing routes a
    /// prover decision on its contents (Rule 16 / [I-44]).
    ///
    /// @return The single shared `DeloadStats` instance.
    DeloadStats& deloadStats();

}
