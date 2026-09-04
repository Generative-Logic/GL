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

#include "deload_stats.hpp"

#include "../infra/diagnostics_log.hpp"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <iostream>

namespace gl {

    namespace {

        // 1 GiB, the byte scale every deload volume is reported in.
        constexpr double kGiB = static_cast<double>(int64_t(1) << 30);

        // Bytes over nanoseconds as GiB/s; zero elapsed time (no traffic)
        // reports 0 — both are defined results of a pure formatting helper,
        // never a prover decision.
        double gibPerSecond(int64_t bytes, int64_t ns) {
            if (ns <= 0) return 0.0;
            return (static_cast<double>(bytes) / kGiB)
                 / (static_cast<double>(ns) / 1e9);
        }

        // Nearest-rank percentile of an already-sorted, non-empty vector.
        int64_t percentile(const std::vector<int64_t>& sorted, double fraction) {
            assert(!sorted.empty());
            std::size_t index = static_cast<std::size_t>(
                fraction * static_cast<double>(sorted.size() - 1) + 0.5);
            if (index >= sorted.size()) index = sorted.size() - 1;
            return sorted[index];
        }

    }

    DeloadStats& deloadStats() {
        static DeloadStats instance;
        return instance;
    }

    void DeloadStats::recordWorkerClaimWait(int phase, int64_t ns) {
        assert(phase >= 1 && phase <= 4
            && "DeloadStats::recordWorkerClaimWait phase out of range "
               "(4 = barrier seam)");
        workerClaimWaitNs[phase].fetch_add(ns, std::memory_order_relaxed);
        workerClaimWaitCount[phase].fetch_add(1, std::memory_order_relaxed);
    }

    void DeloadStats::beginPhaseWindow(int phase) {
        assert(phase >= 1 && phase <= 4
            && "DeloadStats::beginPhaseWindow phase out of range "
               "(4 = barrier seam)");
        assert(phaseWindowId_ == 0
            && "DeloadStats::beginPhaseWindow over an already-open window");
        phaseWindowId_ = phase;
        phaseWindowStart_ = std::chrono::steady_clock::now();
        snapDumpBytes_ = v3DumpBytes.load(std::memory_order_relaxed);
        snapDumpNs_ = v3DumpNs.load(std::memory_order_relaxed);
        snapLoadBytes_ = v3LoadBytes.load(std::memory_order_relaxed);
        snapLoadNs_ = v3LoadNs.load(std::memory_order_relaxed);
        snapRawDumpBytes_ = rawDumpBytes.load(std::memory_order_relaxed);
        snapRawDumpNs_ = rawDumpNs.load(std::memory_order_relaxed);
        snapRawLoadBytes_ = rawLoadBytes.load(std::memory_order_relaxed);
        snapRawLoadNs_ = rawLoadNs.load(std::memory_order_relaxed);
        snapSelfLoadCount_ = workerSelfLoadCount.load(std::memory_order_relaxed);
        snapInlineEvictCount_ =
            workerInlineEvictCount.load(std::memory_order_relaxed);
        snapClaimWaitNs_ =
            workerClaimWaitNs[phase].load(std::memory_order_relaxed);
    }

    PhaseDelta DeloadStats::endPhaseWindow() {
        assert(phaseWindowId_ != 0
            && "DeloadStats::endPhaseWindow without an open window");
        const int phase = phaseWindowId_;
        PhaseDelta d;
        d.phase = phase;
        // Out / in totals fold BOTH streams: v3 canonical (discharge) and
        // v4-raw (eviction). The eviction hot loop is raw, so on the working-
        // set-pager phases the raw deltas dominate.
        d.outBytes =
            (v3DumpBytes.load(std::memory_order_relaxed) - snapDumpBytes_)
            + (rawDumpBytes.load(std::memory_order_relaxed) - snapRawDumpBytes_);
        d.outNs =
            (v3DumpNs.load(std::memory_order_relaxed) - snapDumpNs_)
            + (rawDumpNs.load(std::memory_order_relaxed) - snapRawDumpNs_);
        d.inBytes =
            (v3LoadBytes.load(std::memory_order_relaxed) - snapLoadBytes_)
            + (rawLoadBytes.load(std::memory_order_relaxed) - snapRawLoadBytes_);
        d.inNs =
            (v3LoadNs.load(std::memory_order_relaxed) - snapLoadNs_)
            + (rawLoadNs.load(std::memory_order_relaxed) - snapRawLoadNs_);
        d.selfLoads =
            workerSelfLoadCount.load(std::memory_order_relaxed)
            - snapSelfLoadCount_;
        d.inlineEvicts =
            workerInlineEvictCount.load(std::memory_order_relaxed)
            - snapInlineEvictCount_;
        d.workerWaitNs =
            workerClaimWaitNs[phase].load(std::memory_order_relaxed)
            - snapClaimWaitNs_;
        d.wallSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - phaseWindowStart_).count();
        phaseWindowId_ = 0;

        const bool moved = d.outBytes != 0 || d.inBytes != 0
            || d.workerWaitNs != 0 || d.selfLoads != 0 || d.inlineEvicts != 0;
        if (moved) {
            diagnosticsLog() << "[DELOAD] phase" << phase
                      << ": out " << (static_cast<double>(d.outBytes) / kGiB)
                      << " GiB @ " << gibPerSecond(d.outBytes, d.outNs)
                      << " GiB/s | in "
                      << (static_cast<double>(d.inBytes) / kGiB)
                      << " GiB @ " << gibPerSecond(d.inBytes, d.inNs)
                      << " GiB/s | worker-wait "
                      << (static_cast<double>(d.workerWaitNs) / 1e6)
                      << " ms | self-loads " << d.selfLoads
                      << " | inline-evicts " << d.inlineEvicts << std::endl;
        }
        return d;
    }

    void DeloadStats::reportSummary() const {
        const int64_t dumpBytes = v3DumpBytes.load(std::memory_order_relaxed);
        const int64_t dumpNs = v3DumpNs.load(std::memory_order_relaxed);
        const int64_t loadBytes = v3LoadBytes.load(std::memory_order_relaxed);
        const int64_t loadNs = v3LoadNs.load(std::memory_order_relaxed);
        diagnosticsLog() << "[DELOAD] batch summary:" << std::endl;
        diagnosticsLog() << "  v3 dump: "
                  << v3DumpCount.load(std::memory_order_relaxed) << " ops, "
                  << (static_cast<double>(dumpBytes) / kGiB) << " GiB @ "
                  << gibPerSecond(dumpBytes, dumpNs) << " GiB/s" << std::endl;
        diagnosticsLog() << "  v3 load: "
                  << v3LoadCount.load(std::memory_order_relaxed) << " ops, "
                  << (static_cast<double>(loadBytes) / kGiB) << " GiB @ "
                  << gibPerSecond(loadBytes, loadNs) << " GiB/s" << std::endl;
        const int64_t rDumpBytes = rawDumpBytes.load(std::memory_order_relaxed);
        const int64_t rDumpNs = rawDumpNs.load(std::memory_order_relaxed);
        const int64_t rLoadBytes = rawLoadBytes.load(std::memory_order_relaxed);
        const int64_t rLoadNs = rawLoadNs.load(std::memory_order_relaxed);
        diagnosticsLog() << "  raw dump: "
                  << rawDumpCount.load(std::memory_order_relaxed) << " ops, "
                  << (static_cast<double>(rDumpBytes) / kGiB) << " GiB @ "
                  << gibPerSecond(rDumpBytes, rDumpNs) << " GiB/s" << std::endl;
        diagnosticsLog() << "  raw load: "
                  << rawLoadCount.load(std::memory_order_relaxed) << " ops, "
                  << (static_cast<double>(rLoadBytes) / kGiB) << " GiB @ "
                  << gibPerSecond(rLoadBytes, rLoadNs) << " GiB/s, peak "
                  << (static_cast<double>(
                          rawLoadPeakNs.load(std::memory_order_relaxed)) / 1e6)
                  << " ms" << std::endl;
        diagnosticsLog() << "  index rebuild: "
                  << indexRebuildKeys.load(std::memory_order_relaxed)
                  << " keys, "
                  << (static_cast<double>(
                          indexRebuildNs.load(std::memory_order_relaxed)) / 1e6)
                  << " ms" << std::endl;
        diagnosticsLog() << "  evictions: "
                  << evictCount.load(std::memory_order_relaxed) << " ("
                  << evictBlocks.load(std::memory_order_relaxed)
                  << " blocks, "
                  << (static_cast<double>(
                          evictDumpNs.load(std::memory_order_relaxed)) / 1e6)
                  << " ms) | skipped-clean "
                  << evictSkippedClean.load(std::memory_order_relaxed)
                  << " | tail-dumps "
                  << tailDumpCount.load(std::memory_order_relaxed)
                  << " | full-dumps "
                  << fullDumpCount.load(std::memory_order_relaxed) << std::endl;
        diagnosticsLog() << "  prefetch: issued "
                  << prefetchIssued.load(std::memory_order_relaxed)
                  << " | hit " << prefetchHit.load(std::memory_order_relaxed)
                  << " | miss " << prefetchMiss.load(std::memory_order_relaxed)
                  << " | ring-drops high "
                  << prefetchDroppedFullRing.load(std::memory_order_relaxed)
                  << " low "
                  << evictDroppedFullRing.load(std::memory_order_relaxed)
                  << std::endl;
        diagnosticsLog() << "  worker self-loads: "
                  << workerSelfLoadCount.load(std::memory_order_relaxed) << " ("
                  << (static_cast<double>(
                          workerSelfLoadNs.load(std::memory_order_relaxed))
                      / 1e6)
                  << " ms) | inline-evicts: "
                  << workerInlineEvictCount.load(std::memory_order_relaxed)
                  << " ("
                  << (static_cast<double>(
                          workerInlineEvictNs.load(std::memory_order_relaxed))
                      / 1e6)
                  << " ms) | emergency-evicts: "
                  << emergencyEvictCount.load(std::memory_order_relaxed)
                  << std::endl;
        diagnosticsLog() << "  worker claim-wait: phase1 "
                  << (static_cast<double>(
                          workerClaimWaitNs[1].load(std::memory_order_relaxed))
                      / 1e6)
                  << " ms (" << workerClaimWaitCount[1].load(
                                    std::memory_order_relaxed)
                  << ") | phase2 "
                  << (static_cast<double>(
                          workerClaimWaitNs[2].load(std::memory_order_relaxed))
                      / 1e6)
                  << " ms (" << workerClaimWaitCount[2].load(
                                    std::memory_order_relaxed)
                  << ") | phase3 "
                  << (static_cast<double>(
                          workerClaimWaitNs[3].load(std::memory_order_relaxed))
                      / 1e6)
                  << " ms (" << workerClaimWaitCount[3].load(
                                    std::memory_order_relaxed)
                  << ") | barrier "
                  << (static_cast<double>(
                          workerClaimWaitNs[4].load(std::memory_order_relaxed))
                      / 1e6)
                  << " ms (" << workerClaimWaitCount[4].load(
                                    std::memory_order_relaxed)
                  << ")" << std::endl;
    }

    void DeloadStats::reportLbHistogram(int activeCount,
                                        std::vector<int64_t>& blocksHeld) {
        lbHistogramPrinted_.store(true, std::memory_order_relaxed);
        std::sort(blocksHeld.begin(), blocksHeld.end());
        int64_t maxHeld = 0;
        int64_t p95 = 0;
        int64_t median = 0;
        int64_t belowSmall = 0;
        if (!blocksHeld.empty()) {
            maxHeld = blocksHeld.back();
            p95 = percentile(blocksHeld, 0.95);
            median = percentile(blocksHeld, 0.5);
            for (const int64_t held : blocksHeld)
                if (held < kSmallLbBlocks) ++belowSmall;
        }
        diagnosticsLog() << "[DELOAD] LB blocksHeld: active " << activeCount
                  << " | resident " << blocksHeld.size()
                  << " | max " << maxHeld
                  << " | p95 " << p95
                  << " | median " << median
                  << " | below-" << kSmallLbBlocks << " " << belowSmall
                  << std::endl;
    }

    void DeloadStats::reset() {
        assert(phaseWindowId_ == 0
            && "DeloadStats::reset with a phase window still open");
        v3DumpBytes.store(0, std::memory_order_relaxed);
        v3DumpNs.store(0, std::memory_order_relaxed);
        v3DumpCount.store(0, std::memory_order_relaxed);
        v3LoadBytes.store(0, std::memory_order_relaxed);
        v3LoadNs.store(0, std::memory_order_relaxed);
        v3LoadCount.store(0, std::memory_order_relaxed);
        rawDumpBytes.store(0, std::memory_order_relaxed);
        rawDumpNs.store(0, std::memory_order_relaxed);
        rawDumpCount.store(0, std::memory_order_relaxed);
        rawLoadBytes.store(0, std::memory_order_relaxed);
        rawLoadNs.store(0, std::memory_order_relaxed);
        rawLoadCount.store(0, std::memory_order_relaxed);
        rawLoadPeakNs.store(0, std::memory_order_relaxed);
        indexRebuildNs.store(0, std::memory_order_relaxed);
        indexRebuildKeys.store(0, std::memory_order_relaxed);
        indexRebuildCount.store(0, std::memory_order_relaxed);
        evictSkippedClean.store(0, std::memory_order_relaxed);
        tailDumpCount.store(0, std::memory_order_relaxed);
        fullDumpCount.store(0, std::memory_order_relaxed);
        evictCount.store(0, std::memory_order_relaxed);
        evictBlocks.store(0, std::memory_order_relaxed);
        evictDumpNs.store(0, std::memory_order_relaxed);
        prefetchIssued.store(0, std::memory_order_relaxed);
        prefetchHit.store(0, std::memory_order_relaxed);
        prefetchMiss.store(0, std::memory_order_relaxed);
        prefetchDroppedFullRing.store(0, std::memory_order_relaxed);
        evictDroppedFullRing.store(0, std::memory_order_relaxed);
        workerSelfLoadNs.store(0, std::memory_order_relaxed);
        workerSelfLoadCount.store(0, std::memory_order_relaxed);
        workerInlineEvictNs.store(0, std::memory_order_relaxed);
        workerInlineEvictCount.store(0, std::memory_order_relaxed);
        emergencyEvictCount.store(0, std::memory_order_relaxed);
        for (int p = 0; p < 5; ++p) {
            workerClaimWaitNs[p].store(0, std::memory_order_relaxed);
            workerClaimWaitCount[p].store(0, std::memory_order_relaxed);
        }
        lbHistogramPrinted_.store(false, std::memory_order_relaxed);
    }

}
