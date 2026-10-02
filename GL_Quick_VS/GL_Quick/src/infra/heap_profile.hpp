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
#include <mutex>
#include <ostream>
#include <string>
#include <vector>

namespace gl {

    /// @brief The heap side of the memory census: exact live C++ heap bytes
    ///        and a sampled profile of the allocation sites holding them.
    ///
    /// @details
    /// The static pools report their own contents; the malloc heap outside
    /// them was visible only as mimalloc's committed bytes, which cannot tell
    /// live data from freed memory the allocator retains, nor name the owner.
    /// Two instruments close that gap (measurement builds only,
    /// `MEM_MEASUREMENT`):
    ///
    /// - **Live bytes.** The replaced global `operator new` / `operator delete`
    ///   (`heap_profile_hooks.cpp`, mimalloc builds) add and subtract each
    ///   block's usable size in thread-sharded counters; `liveBytes()` sums
    ///   them. Committed heap minus live heap is the allocator's retention.
    ///   Allocations made through `malloc` (C libraries, the CUDA runtime) are
    ///   outside this count.
    /// - **Allocation sites (opt-in, `GL_HEAP_PROFILE_KIB`).** With `enable`,
    ///   every thread samples one allocation per `interval` bytes it allocates
    ///   and every allocation of at least `interval` bytes, records the call
    ///   stack with the sample's weight (the larger of its size and the
    ///   interval), and removes the sample when the block is freed. The live
    ///   weights per call stack estimate which code holds the live heap; the
    ///   census snapshots them at the batch's heap peak and at its largest
    ///   end-of-iteration barrier, and `writeReport` names the top sites.
    ///
    /// @invariant Nothing here is a proof input (Rule 16): no prover, deload or
    ///            steward decision reads a heap-profile value.
    /// @invariant The allocation and free paths never allocate: every table
    ///            lives in storage reserved once by `enable`.
    /// @see `memory_census`, `docs/agentic_swdd/_meta/memory_measurement.md`.
    namespace heap_profile {

        /// @brief Frames kept per sampled call stack.
        ///
        /// @details
        /// Enough for the allocator's own frames plus the five program frames
        /// a label shows; deeper frames would only split one site into many
        /// (recursion depths, thread entry paths).
        inline constexpr int32_t kSiteFrames = 16;

        /// @brief Add `delta` to the live-heap counter of the calling thread's
        ///        shard.
        ///
        /// @details
        /// Sixty-four cache-line-separated atomic shards, one chosen per thread
        /// on its first call, keep concurrent allocation and free paths from
        /// contending on one counter. Relaxed ordering: the sum is read only at
        /// the census's quiescent seams.
        ///
        /// @param delta Signed byte change (usable size of the block).
        void addLiveBytes(int64_t delta);

        /// @brief Live C++ heap bytes: the sum of all shards.
        ///
        /// @return Usable bytes of every block allocated and not yet freed
        ///         through the replaced global operators since process start.
        int64_t liveBytes();

        /// @brief Whether this build replaces the global allocation operators
        ///        and therefore counts live bytes.
        ///
        /// @return `true` in measurement builds that link mimalloc.
        bool countsNewDelete();

        /// @brief One distinct sampled call stack and the sampled bytes it holds.
        ///
        /// @details
        /// `hash` is the stack hash (0 marks an empty slot); `frames` holds
        /// `depth` return addresses, innermost first; `liveBytes` is the sum of
        /// the weights of the site's live samples and `liveCount` their number.
        struct Site {
            uint64_t hash = 0;
            int32_t depth = 0;
            void* frames[kSiteFrames] = {};
            int64_t liveBytes = 0;
            int64_t liveCount = 0;
        };

        /// @brief One row of a site snapshot.
        ///
        /// @details
        /// `site` indexes the profile's site table; `bytes` and `count` are the
        /// site's live bytes and live samples at the snapshot.
        struct SiteBytes {
            int32_t site = 0;
            int64_t bytes = 0;
            int64_t count = 0;
        };

        /// @brief Fixed-capacity table of live samples grouped by call stack.
        ///
        /// @details
        /// Two open-addressing tables in storage reserved once by `init`: the
        /// sites (keyed by stack hash, insert-only) and the live samples (keyed
        /// by block address, linear probing with backward-shift deletion, so no
        /// tombstones accumulate over a batch's churn). A counting filter over
        /// block addresses lets the free path skip the lock for every block that
        /// cannot be a sample. All table operations take one internal mutex; the
        /// filter check does not. Full tables are an assert naming the capacity
        /// (Rule 19).
        class SiteProfile {
        public:
            SiteProfile() = default;

            /// @brief Returns the reserved storage.
            ~SiteProfile();

            SiteProfile(const SiteProfile&) = delete;
            SiteProfile& operator=(const SiteProfile&) = delete;

            /// @brief Reserve and commit the three tables.
            ///
            /// @param siteCapacity   Site slots; a power of two.
            /// @param sampleCapacity Sample slots; a power of two.
            /// @param filterCapacity Filter counters; a power of two.
            void init(int32_t siteCapacity, int32_t sampleCapacity, int32_t filterCapacity);

            /// @brief Whether `init` has run.
            ///
            /// @return `true` after `init`.
            bool initialized() const;

            /// @brief Record one live sample under its call stack.
            ///
            /// @details
            /// Finds or inserts the site by `stackHash` (a colliding hash with
            /// different frames is an assert), inserts the sample (a second
            /// live sample at one address is an assert), and adds `weight` to
            /// the site.
            ///
            /// @param ptr       Sampled block address; not null.
            /// @param weight    Bytes the sample stands for; positive.
            /// @param frames    Call stack, innermost first.
            /// @param depth     Frames in `frames`; 1..`kSiteFrames`.
            /// @param stackHash Hash of the frames; not zero.
            void record(const void* ptr, int64_t weight, void* const* frames,
                        int32_t depth, uint64_t stackHash);

            /// @brief Lock-free pre-check of the free path.
            ///
            /// @param ptr Block address being freed.
            /// @return `false` when `ptr` is certainly not a live sample.
            bool mayHold(const void* ptr) const;

            /// @brief Remove the live sample at `ptr`, if any.
            ///
            /// @details
            /// Both outcomes are part of the contract: the filter admits
            /// addresses that share a counter with a live sample.
            ///
            /// @param ptr Block address being freed.
            /// @return `true` when a sample was removed.
            bool forget(const void* ptr);

            /// @brief Copy every site with live samples, largest first.
            ///
            /// @param out      Destination rows.
            /// @param capacity Rows available in `out`.
            /// @return Rows written (at most `capacity`).
            int32_t snapshot(SiteBytes* out, int32_t capacity) const;

            /// @brief One site by index.
            ///
            /// @param index A `SiteBytes::site` value.
            /// @return The site.
            const Site& site(int32_t index) const;

            /// @brief Σ live sample weights.
            ///
            /// @return The sampled estimate of the live heap.
            int64_t sampledLiveBytes() const;

            /// @brief Number of live samples.
            ///
            /// @return Samples recorded and not yet forgotten.
            int64_t liveSamples() const;

        private:
            struct Sample {
                uintptr_t ptr;      // 0 = empty slot
                int64_t weight;
                int32_t site;
                int32_t pad;
            };

            int32_t findSiteLocked(uint64_t stackHash, void* const* frames, int32_t depth);
            uint64_t filterIndex(uintptr_t ptr) const;

            mutable std::mutex mutex_;
            Site* sites_ = nullptr;
            Sample* samples_ = nullptr;
            std::atomic<uint16_t>* filter_ = nullptr;
            int32_t siteCapacity_ = 0;
            int32_t sampleCapacity_ = 0;
            int32_t filterCapacity_ = 0;
            int32_t siteCount_ = 0;
            int64_t sampleCount_ = 0;
            int64_t sampledBytes_ = 0;
            int64_t storageBytes_ = 0;
            char* storage_ = nullptr;
        };

        /// @brief Start sampling allocation sites for the rest of the process.
        ///
        /// @details
        /// Reserves the process-wide profile and its two snapshots once; later
        /// calls assert. Called from `main` when `GL_HEAP_PROFILE_KIB` is set
        /// (batch runs only; `--unit-tests` leaves it to the heap-profile
        /// tests).
        ///
        /// @param intervalBytes Sampling interval; positive.
        void enable(int64_t intervalBytes);

        /// @brief Whether `enable` has run.
        ///
        /// @return `true` while allocation sites are sampled.
        bool enabled();

        /// @brief Allocation hook: count the block and maybe sample it.
        ///
        /// @param ptr    The new block.
        /// @param usable Its usable size.
        void onAllocate(void* ptr, int64_t usable);

        /// @brief Free hook: uncount the block and forget its sample.
        ///
        /// @param ptr    The block about to be freed.
        /// @param usable Its usable size.
        void onFree(void* ptr, int64_t usable);

        /// @brief Census hook at every process sample: keep the site snapshots
        ///        of the heap peak and of the largest barrier.
        ///
        /// @param live        `liveBytes()` at the sample.
        /// @param barrierStep Whether the sample closes an end-of-iteration
        ///                    barrier.
        void noteCensusSample(int64_t live, bool barrierStep);

        /// @brief Readable label of one call stack.
        ///
        /// @details
        /// Drops the allocator's own frames (the replaced operators, this
        /// namespace, mimalloc, the C runtime), keeps the last standard-library
        /// frame before the first program frame as the container operation, and
        /// joins up to five program frames innermost first with " < " ("called
        /// from"). Each name is cut at 110 characters.
        ///
        /// @param names Symbol names of the frames, innermost first.
        /// @return The label.
        std::string siteLabel(const std::vector<std::string>& names);

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
        std::string symbolName(void* address);

        /// @brief Print the heap-site sections for this batch.
        ///
        /// @details
        /// Writes nothing when sampling is off. Otherwise one markdown table per
        /// snapshot (the heap peak and the largest barrier) with the top sites'
        /// sampled live bytes and labels, and one `[heap-site]` diagnostics line
        /// per row.
        ///
        /// @param diagnostics The diagnostics log stream.
        /// @param markdown    The run's memory log stream.
        void writeReport(std::ostream& diagnostics, std::ostream& markdown);

    } // namespace heap_profile
} // namespace gl
