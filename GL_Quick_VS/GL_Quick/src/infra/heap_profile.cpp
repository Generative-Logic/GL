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

#include "heap_profile.hpp"

#include "../memory_infra/address_space.hpp"

#include <algorithm>
#include <atomic>
#include <cassert>
#include <cstring>
#include <iomanip>
#include <new>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <dbghelp.h>
#pragma comment(lib, "dbghelp.lib")
#else
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <unwind.h>
#include <cstdlib>
#endif

namespace gl {
    namespace heap_profile {

        namespace {

            /// @brief One cache line holding one live-counter shard.
            struct alignas(64) LiveShard {
                std::atomic<int64_t> bytes{ 0 };
            };

            constexpr int32_t kLiveShards = 64;
            LiveShard gLiveShards[kLiveShards];
            std::atomic<int32_t> gNextShard{ 0 };
            thread_local int32_t tlsShard = -1;

            // Sampling state. `gProfile` is placed in reserved storage by
            // `enable` and never destroyed, so late frees at process exit
            // still find it.
            std::atomic<int64_t> gIntervalBytes{ 0 };
            SiteProfile* gProfile = nullptr;
            thread_local int64_t tlsSinceSample = 0;
            thread_local bool tlsInProfiler = false;

            /// @brief The two census snapshots of the site table.
            struct Snapshot {
                SiteBytes* rows = nullptr;
                int32_t count = 0;
                int64_t live = -1;          // liveBytes() when taken; -1 = none
                int64_t sampled = 0;        // sampledLiveBytes() when taken
            };
            Snapshot gPeak;
            Snapshot gBarrier;
            int32_t gSnapshotCapacity = 0;
            std::mutex gSnapshotMutex;

            constexpr int32_t kSiteCapacity = 1 << 18;
            constexpr int32_t kSampleCapacity = 1 << 20;
            constexpr int32_t kFilterCapacity = 1 << 23;

            /// @brief Mix a 64-bit value (the splitmix64 finalizer).
            ///
            /// @param x Input.
            /// @return Well-spread hash of `x`.
            uint64_t mix(uint64_t x) {
                x ^= x >> 30;
                x *= 0xbf58476d1ce4e5b9ULL;
                x ^= x >> 27;
                x *= 0x94d049bb133111ebULL;
                x ^= x >> 31;
                return x;
            }

            /// @brief Capture the calling thread's stack, innermost first.
            ///
            /// @param frames Destination (`kSiteFrames` entries).
            /// @param hash   Receives a non-zero hash of the captured frames.
            /// @return Frames captured.
            int32_t captureStack(void** frames, uint64_t* hash) {
#ifdef _WIN32
                const int32_t depth = static_cast<int32_t>(
                    RtlCaptureStackBackTrace(0, kSiteFrames, frames, nullptr));
#else
                const int32_t depth = backtrace(frames, kSiteFrames);
#endif
                assert(depth > 0);
                uint64_t h = 0x9e3779b97f4a7c15ULL;
                for (int32_t i = 0; i < depth; ++i)
                    h = mix(h ^ reinterpret_cast<uintptr_t>(frames[i]));
                *hash = h == 0 ? 1 : h;
                return depth;
            }

            /// @brief Copy the profile's live sites into `snap`.
            ///
            /// @param snap  Destination snapshot.
            /// @param live  `liveBytes()` at the sample.
            void takeSnapshot(Snapshot& snap, int64_t live) {
                snap.count = gProfile->snapshot(snap.rows, gSnapshotCapacity);
                snap.live = live;
                snap.sampled = gProfile->sampledLiveBytes();
            }

        } // namespace

        /// @brief Add `delta` to the live-heap counter of the calling thread's
        ///        shard.
        ///
        /// @param delta Signed byte change (usable size of the block).
        void addLiveBytes(int64_t delta) {
            if (tlsShard < 0)
                tlsShard = gNextShard.fetch_add(1, std::memory_order_relaxed) % kLiveShards;
            gLiveShards[tlsShard].bytes.fetch_add(delta, std::memory_order_relaxed);
        }

        /// @brief Live C++ heap bytes: the sum of all shards.
        ///
        /// @return Usable bytes of every block allocated and not yet freed
        ///         through the replaced global operators since process start.
        int64_t liveBytes() {
            int64_t total = 0;
            for (const LiveShard& shard : gLiveShards)
                total += shard.bytes.load(std::memory_order_relaxed);
            return total;
        }

        /// @brief Returns the reserved storage.
        SiteProfile::~SiteProfile() {
            if (storage_ != nullptr) releaseAddressSpace(storage_, storageBytes_);
        }

        /// @brief Reserve and commit the three tables.
        ///
        /// @param siteCapacity   Site slots; a power of two.
        /// @param sampleCapacity Sample slots; a power of two.
        /// @param filterCapacity Filter counters; a power of two.
        void SiteProfile::init(int32_t siteCapacity, int32_t sampleCapacity,
                               int32_t filterCapacity) {
            assert(storage_ == nullptr);
            const auto pow2 = [](int32_t v) { return v > 0 && (v & (v - 1)) == 0; };
            assert(pow2(siteCapacity) && pow2(sampleCapacity) && pow2(filterCapacity));
            const int64_t siteBytes = int64_t(siteCapacity) * int64_t(sizeof(Site));
            const int64_t sampleBytes = int64_t(sampleCapacity) * int64_t(sizeof(Sample));
            const int64_t filterBytes = int64_t(filterCapacity) * int64_t(sizeof(std::atomic<uint16_t>));
            storageBytes_ = siteBytes + sampleBytes + filterBytes;
            storage_ = reserveAddressSpace(storageBytes_);
            commitAddressSpace(storage_, storageBytes_);
            sites_ = reinterpret_cast<Site*>(storage_);
            samples_ = reinterpret_cast<Sample*>(storage_ + siteBytes);
            filter_ = reinterpret_cast<std::atomic<uint16_t>*>(storage_ + siteBytes + sampleBytes);
            for (int32_t i = 0; i < filterCapacity; ++i)
                new (&filter_[i]) std::atomic<uint16_t>(0);
            siteCapacity_ = siteCapacity;
            sampleCapacity_ = sampleCapacity;
            filterCapacity_ = filterCapacity;
        }

        /// @brief Whether `init` has run.
        ///
        /// @return `true` after `init`.
        bool SiteProfile::initialized() const {
            return storage_ != nullptr;
        }

        /// @brief The filter counter of a block address.
        ///
        /// @param ptr Block address.
        /// @return Counter index.
        uint64_t SiteProfile::filterIndex(uintptr_t ptr) const {
            return mix(ptr) & uint64_t(filterCapacity_ - 1);
        }

        /// @brief Find or insert the site of a call stack. Caller holds
        ///        `mutex_`.
        ///
        /// @param stackHash Hash of the frames; not zero.
        /// @param frames    Call stack, innermost first.
        /// @param depth     Frames in `frames`.
        /// @return The site index.
        int32_t SiteProfile::findSiteLocked(uint64_t stackHash, void* const* frames,
                                            int32_t depth) {
            const uint64_t mask = uint64_t(siteCapacity_ - 1);
            for (uint64_t i = mix(stackHash) & mask;; i = (i + 1) & mask) {
                Site& s = sites_[i];
                if (s.hash == stackHash) {
                    assert(s.depth == depth
                        && std::memcmp(s.frames, frames, sizeof(void*) * depth) == 0
                        && "heap profile: two call stacks share one hash");
                    return static_cast<int32_t>(i);
                }
                if (s.hash == 0) {
                    assert(int64_t(siteCount_ + 1) * 4 <= int64_t(siteCapacity_) * 3
                        && "heap profile: site table full — raise kSiteCapacity");
                    s.hash = stackHash;
                    s.depth = depth;
                    std::memcpy(s.frames, frames, sizeof(void*) * depth);
                    ++siteCount_;
                    return static_cast<int32_t>(i);
                }
            }
        }

        /// @brief Record one live sample under its call stack.
        ///
        /// @param ptr       Sampled block address; not null.
        /// @param weight    Bytes the sample stands for; positive.
        /// @param frames    Call stack, innermost first.
        /// @param depth     Frames in `frames`; 1..`kSiteFrames`.
        /// @param stackHash Hash of the frames; not zero.
        void SiteProfile::record(const void* ptr, int64_t weight, void* const* frames,
                                 int32_t depth, uint64_t stackHash) {
            assert(ptr != nullptr && weight > 0 && depth > 0 && depth <= kSiteFrames
                && stackHash != 0);
            const uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
            std::lock_guard<std::mutex> lock(mutex_);
            assert((sampleCount_ + 1) * 4 <= int64_t(sampleCapacity_) * 3
                && "heap profile: sample table full — raise kSampleCapacity");
            const int32_t site = findSiteLocked(stackHash, frames, depth);
            const uint64_t mask = uint64_t(sampleCapacity_ - 1);
            uint64_t i = mix(key) & mask;
            while (samples_[i].ptr != 0) {
                assert(samples_[i].ptr != key && "heap profile: address sampled twice");
                i = (i + 1) & mask;
            }
            samples_[i] = Sample{ key, weight, site, 0 };
            ++sampleCount_;
            sampledBytes_ += weight;
            sites_[site].liveBytes += weight;
            ++sites_[site].liveCount;
            // Relaxed publication suffices: a block reaches its free only after
            // the program has passed the pointer on through its own
            // synchronization, which orders this increment before that load.
            const uint16_t before =
                filter_[filterIndex(key)].fetch_add(1, std::memory_order_relaxed);
            assert(before < UINT16_MAX);
        }

        /// @brief Lock-free pre-check of the free path.
        ///
        /// @param ptr Block address being freed.
        /// @return `false` when `ptr` is certainly not a live sample.
        bool SiteProfile::mayHold(const void* ptr) const {
            const uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
            return filter_[filterIndex(key)].load(std::memory_order_relaxed) != 0;
        }

        /// @brief Remove the live sample at `ptr`, if any.
        ///
        /// @param ptr Block address being freed.
        /// @return `true` when a sample was removed.
        bool SiteProfile::forget(const void* ptr) {
            const uintptr_t key = reinterpret_cast<uintptr_t>(ptr);
            std::lock_guard<std::mutex> lock(mutex_);
            const uint64_t mask = uint64_t(sampleCapacity_ - 1);
            uint64_t i = mix(key) & mask;
            while (samples_[i].ptr != key) {
                if (samples_[i].ptr == 0) return false;
                i = (i + 1) & mask;
            }
            const Sample gone = samples_[i];
            sites_[gone.site].liveBytes -= gone.weight;
            --sites_[gone.site].liveCount;
            sampledBytes_ -= gone.weight;
            --sampleCount_;
            const uint16_t before =
                filter_[filterIndex(key)].fetch_sub(1, std::memory_order_relaxed);
            assert(before > 0);
            // Backward-shift deletion keeps every probe chain intact without
            // tombstones.
            uint64_t hole = i;
            for (uint64_t j = (i + 1) & mask; samples_[j].ptr != 0; j = (j + 1) & mask) {
                const uint64_t home = mix(samples_[j].ptr) & mask;
                const bool stays = hole <= j ? (hole < home && home <= j)
                                             : (hole < home || home <= j);
                if (stays) continue;
                samples_[hole] = samples_[j];
                hole = j;
            }
            samples_[hole] = Sample{ 0, 0, 0, 0 };
            return true;
        }

        /// @brief Copy every site with live samples, largest first.
        ///
        /// @param out      Destination rows.
        /// @param capacity Rows available in `out`.
        /// @return Rows written (at most `capacity`).
        int32_t SiteProfile::snapshot(SiteBytes* out, int32_t capacity) const {
            int32_t written = 0;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                for (int32_t i = 0; i < siteCapacity_ && written < capacity; ++i) {
                    const Site& s = sites_[i];
                    if (s.hash == 0 || s.liveCount == 0) continue;
                    out[written++] = SiteBytes{ i, s.liveBytes, s.liveCount };
                }
            }
            std::sort(out, out + written, [](const SiteBytes& a, const SiteBytes& b) {
                return a.bytes != b.bytes ? a.bytes > b.bytes : a.site < b.site;
            });
            return written;
        }

        /// @brief One site by index.
        ///
        /// @param index A `SiteBytes::site` value.
        /// @return The site.
        const Site& SiteProfile::site(int32_t index) const {
            assert(index >= 0 && index < siteCapacity_ && sites_[index].hash != 0);
            return sites_[index];
        }

        /// @brief Σ live sample weights.
        ///
        /// @return The sampled estimate of the live heap.
        int64_t SiteProfile::sampledLiveBytes() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return sampledBytes_;
        }

        /// @brief Number of live samples.
        ///
        /// @return Samples recorded and not yet forgotten.
        int64_t SiteProfile::liveSamples() const {
            std::lock_guard<std::mutex> lock(mutex_);
            return sampleCount_;
        }

        /// @brief Start sampling allocation sites for the rest of the process.
        ///
        /// @param intervalBytes Sampling interval; positive.
        void enable(int64_t intervalBytes) {
            assert(intervalBytes > 0 && gProfile == nullptr);
            char* const place = reserveAddressSpace(int64_t(sizeof(SiteProfile)));
            commitAddressSpace(place, int64_t(sizeof(SiteProfile)));
            gProfile = new (place) SiteProfile();
            gProfile->init(kSiteCapacity, kSampleCapacity, kFilterCapacity);
            gSnapshotCapacity = kSiteCapacity;
            const int64_t rowBytes = int64_t(kSiteCapacity) * int64_t(sizeof(SiteBytes));
            char* const rows = reserveAddressSpace(2 * rowBytes);
            commitAddressSpace(rows, 2 * rowBytes);
            gPeak.rows = reinterpret_cast<SiteBytes*>(rows);
            gBarrier.rows = reinterpret_cast<SiteBytes*>(rows + rowBytes);
            gIntervalBytes.store(intervalBytes, std::memory_order_release);
        }

        /// @brief Whether `enable` has run.
        ///
        /// @return `true` while allocation sites are sampled.
        bool enabled() {
            return gIntervalBytes.load(std::memory_order_acquire) != 0;
        }

        /// @brief Allocation hook: count the block and maybe sample it.
        ///
        /// @param ptr    The new block.
        /// @param usable Its usable size.
        void onAllocate(void* ptr, int64_t usable) {
            addLiveBytes(usable);
            const int64_t interval = gIntervalBytes.load(std::memory_order_relaxed);
            if (interval == 0 || tlsInProfiler) return;
            int64_t weight = 0;
            if (usable >= interval) {
                weight = usable;
            }
            else {
                tlsSinceSample += usable;
                if (tlsSinceSample < interval) return;
                tlsSinceSample = 0;
                weight = interval;
            }
            tlsInProfiler = true;
            void* frames[kSiteFrames];
            uint64_t hash = 0;
            const int32_t depth = captureStack(frames, &hash);
            gProfile->record(ptr, weight, frames, depth, hash);
            tlsInProfiler = false;
        }

        /// @brief Free hook: uncount the block and forget its sample.
        ///
        /// @param ptr    The block about to be freed.
        /// @param usable Its usable size.
        void onFree(void* ptr, int64_t usable) {
            addLiveBytes(-usable);
            if (gIntervalBytes.load(std::memory_order_relaxed) == 0) return;
            if (gProfile->mayHold(ptr)) gProfile->forget(ptr);
        }

        /// @brief Census hook at every process sample.
        ///
        /// @param live        `liveBytes()` at the sample.
        /// @param barrierStep Whether the sample closes an end-of-iteration
        ///                    barrier.
        void noteCensusSample(int64_t live, bool barrierStep) {
            if (!enabled()) return;
            std::lock_guard<std::mutex> lock(gSnapshotMutex);
            if (live > gPeak.live) takeSnapshot(gPeak, live);
            if (barrierStep && live > gBarrier.live) takeSnapshot(gBarrier, live);
        }

        /// @brief Readable label of one call stack.
        ///
        /// @param names Symbol names of the frames, innermost first.
        /// @return The label.
        std::string siteLabel(const std::vector<std::string>& names) {
            const auto isAllocator = [](const std::string& n) {
                return n.empty() || n.find("heap_profile") != std::string::npos
                    || n.find("operator new") != std::string::npos
                    || n.rfind("mi_", 0) == 0 || n.rfind("_mi_", 0) == 0
                    || n == "malloc" || n.rfind("_malloc", 0) == 0;
            };
            const auto isStd = [](const std::string& n) {
                return n.rfind("std::", 0) == 0 || n.rfind("gl::ScratchVector", 0) == 0;
            };
            const auto cut = [](const std::string& n) {
                return n.size() <= 110 ? n : n.substr(0, 107) + "...";
            };
            std::string container;
            std::string chain;
            int32_t kept = 0;
            for (const std::string& n : names) {
                if (isAllocator(n)) continue;
                if (kept == 0 && isStd(n)) {
                    container = n;
                    continue;
                }
                if (kept == 5) break;
                if (kept > 0) chain += " < ";
                chain += cut(n);
                ++kept;
            }
            if (!container.empty()) chain += " [" + cut(container) + "]";
            return chain;
        }

        /// @brief Symbol name of the function that contains one code address
        ///        (empty when unresolved).
        ///
        /// @details
        /// The platform lookups answer with the nearest symbol at or below the
        /// address. In an executable without a PDB, DbgHelp sees only the
        /// exported functions, and `dladdr` sees only the dynamic symbol table,
        /// so an address inside an unlisted function would carry the name of
        /// an unrelated listed function that precedes it. A name is therefore
        /// returned only when the address lies inside the named function: at
        /// the symbol itself, within the symbol's size when the PDB records
        /// one, or in the function whose start the unwind tables report
        /// (`RtlLookupFunctionEntry` on Windows, `_Unwind_FindEnclosingFunction`
        /// elsewhere) when that start is the symbol's address.
        ///
        /// @param address A return address or function address in this process.
        /// @return The undecorated name of the function that contains `address`,
        ///         or an empty string when no symbol is known to cover it (both
        ///         outcomes are part of the contract).
        std::string symbolName(void* address) {
#ifdef _WIN32
            static std::mutex dbghelpMutex;   // DbgHelp is single-threaded
            static bool ready = false;
            std::lock_guard<std::mutex> lock(dbghelpMutex);
            if (!ready) {
                SymSetOptions(SYMOPT_UNDNAME | SYMOPT_DEFERRED_LOADS);
                const BOOL ok = SymInitialize(GetCurrentProcess(), nullptr, TRUE);
                assert(ok && "heap profile: SymInitialize failed");
                ready = true;
            }
            alignas(SYMBOL_INFO) char buffer[sizeof(SYMBOL_INFO) + 512];
            SYMBOL_INFO* info = reinterpret_cast<SYMBOL_INFO*>(buffer);
            info->SizeOfStruct = sizeof(SYMBOL_INFO);
            info->MaxNameLen = 511;
            const DWORD64 target = reinterpret_cast<DWORD64>(address);
            DWORD64 displacement = 0;
            if (!SymFromAddr(GetCurrentProcess(), target, &displacement, info))
                return std::string();
            bool inside = displacement == 0;
            if (!inside && info->Size != 0) {
                inside = displacement < info->Size;
            } else if (!inside) {
                DWORD64 imageBase = 0;
                const PRUNTIME_FUNCTION entry =
                    RtlLookupFunctionEntry(target, &imageBase, nullptr);
                inside = entry != nullptr
                    && imageBase + entry->BeginAddress == info->Address;
            }
            if (!inside) return std::string();
            return std::string(info->Name, info->NameLen);
#else
            Dl_info info{};
            if (dladdr(address, &info) == 0 || info.dli_sname == nullptr)
                return std::string();
            if (info.dli_saddr != address
                && info.dli_saddr != _Unwind_FindEnclosingFunction(address))
                return std::string();
            int status = 0;
            char* demangled = abi::__cxa_demangle(info.dli_sname, nullptr, nullptr, &status);
            std::string name = status == 0 ? std::string(demangled) : std::string(info.dli_sname);
            std::free(demangled);
            return name;
#endif
        }

        /// @brief Print the heap-site sections for this batch.
        ///
        /// @param diagnostics The diagnostics log stream.
        /// @param markdown    The run's memory log stream.
        void writeReport(std::ostream& diagnostics, std::ostream& markdown) {
            if (!enabled()) return;
            constexpr int32_t kTop = 30;
            const auto mib = [](int64_t bytes) { return double(bytes) / (1024.0 * 1024.0); };
            std::lock_guard<std::mutex> lock(gSnapshotMutex);
            const Snapshot* const snaps[] = { &gPeak, &gBarrier };
            const char* const titles[] = { "at the heap peak", "at the largest barrier" };
            const char* const keys[] = { "peak", "barrier" };
            for (int k = 0; k < 2; ++k) {
                const Snapshot& snap = *snaps[k];
                if (snap.live < 0) continue;
                markdown << "\n### Heap live by allocation site " << titles[k]
                         << " (sampled every " << gIntervalBytes.load() / 1024
                         << " KiB, MiB)\n\n"
                         << "Live C++ heap " << std::fixed << std::setprecision(1)
                         << mib(snap.live) << " MiB; sampled estimate "
                         << mib(snap.sampled) << " MiB over " << snap.count
                         << " call stacks. Rows are call stacks, innermost program "
                            "frame first (`a < b` = a called from b); the bracket names "
                            "the container operation.\n\n"
                         << "| # | MiB | Share | Samples | Site |\n|---:|---:|---:|---:|---|\n";
                for (int32_t r = 0; r < snap.count && r < kTop; ++r) {
                    const SiteBytes& row = snap.rows[r];
                    const Site& site = gProfile->site(row.site);
                    std::vector<std::string> names;
                    for (int32_t f = 0; f < site.depth; ++f)
                        names.push_back(symbolName(site.frames[f]));
                    const std::string label = siteLabel(names);
                    const double share = snap.sampled > 0
                        ? 100.0 * double(row.bytes) / double(snap.sampled) : 0.0;
                    diagnostics << "[heap-site] snapshot=" << keys[k] << " rank=" << (r + 1)
                                << " bytes=" << row.bytes << " samples=" << row.count
                                << " site=" << label << std::endl;
                    markdown << "| " << (r + 1) << " | " << std::fixed
                             << std::setprecision(1) << mib(row.bytes) << " | "
                             << share << "% | " << row.count << " | `" << label << "` |\n";
                }
            }
        }

    } // namespace heap_profile
} // namespace gl
