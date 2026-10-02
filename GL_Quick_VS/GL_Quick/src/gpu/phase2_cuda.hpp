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

#include "phase2_projection.hpp"

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include "phase2_streaming.hpp"
#include "../memory_infra/paged_vector.hpp"

namespace gl::gpu {

    /// @brief Device payload and exact aligned arena footprint before allocation.
    ///
    /// @details
    /// Library workspace is included. Arena bytes assume an initially aligned
    /// cursor and include all padding between this owner's arrays.
    struct Phase2DeviceAllocationSize {
        uint64_t deviceBytes{ 0 };
        uint64_t arenaBytes{ 0 };
    };

    /// @brief One live contiguous lease of physical GPU allocation blocks.
    struct Phase2GpuLease {
        uint32_t first{ 0 };
        uint32_t count{ 0 };
        uint64_t generation{ 0 };
    };

    /// @brief Own one automatically sized, reusable GPU block reservation.
    ///
    /// @details
    /// Host metadata is allocated once for the card's block count. Leases name
    /// physical storage only; no proof identity or ordering depends on a lease.
    /// A scheduler checks free runs before reserving and drains when full.
    class CudaPhase2BlockPool {
        Phase2GpuBudget budget_{};
        void* device_{ nullptr };
        uint64_t deviceTotalBytes_{ 0 };        // cudaMemGetInfo total at construction
        uint64_t deviceFreeBeforePoolBytes_{ 0 };  // cudaMemGetInfo free before the reservation
        LbArena metadataArena_;
        DirtyState metadataDirty_{ DirtyState::Clean };
        PagedVector<uint64_t> owners_;
        uint32_t usedBlocks_{ 0 };

    public:
        /// @brief Reserve the automatic GPU budget, optionally capped for tests.
        ///
        /// @details
        /// Initializes CUDA before measuring free memory and allocates one
        /// contiguous reservation with the common runtime reserve left outside.
        ///
        /// @param limitBytes Optional working-byte ceiling, zero for automatic.
        /// @return An empty pool ready for block leases.
        /// @invariant Device allocation and metadata sizes remain fixed.
        explicit CudaPhase2BlockPool(uint64_t limitBytes = 0);

        /// @brief Release the reservation after every lease has been returned.
        ///
        /// @details
        /// Outstanding leases at teardown are a lifetime contract violation.
        ///
        /// @return Nothing.
        /// @invariant Every lease owner is destroyed before the pool.
        ~CudaPhase2BlockPool();
        CudaPhase2BlockPool(const CudaPhase2BlockPool&) = delete;
        CudaPhase2BlockPool& operator=(const CudaPhase2BlockPool&) = delete;

        /// @brief Read the fixed allocation and runtime-reserve census.
        ///
        /// @details
        /// This is resource metadata, independent of proof state.
        ///
        /// @return The pool's immutable byte and block budget.
        const Phase2GpuBudget& budget() const { return budget_; }

        /// @brief Device memory the driver reported as total when the pool
        ///        was built — the VRAM denominator of the memory report.
        ///
        /// @details
        /// Read once in the constructor through `cudaMemGetInfo` after the
        /// runtime context exists, before the pool's own reservation.
        /// Observability only (Rule 16).
        ///
        /// @return Total device bytes.
        uint64_t deviceTotalBytes() const { return deviceTotalBytes_; }

        /// @brief Device memory the driver reported as free just before the
        ///        pool's reservation — total minus the CUDA context and any
        ///        other resident user of the card.
        ///
        /// @details
        /// Together with `deviceTotalBytes` and `budget().bytes` it splits the
        /// card into: context and other users (`total - freeBefore`), this
        /// process's fixed Phase 2 reservation (`budget().bytes`), and the
        /// runtime reserve left free (`freeBefore - budget().bytes`).
        /// Observability only (Rule 16).
        ///
        /// @return Free device bytes before the reservation.
        uint64_t deviceFreeBeforePoolBytes() const { return deviceFreeBeforePoolBytes_; }

        /// @brief Count blocks currently available for lending.
        ///
        /// @details
        /// Zero is the defined full-pool state used by the scheduler.
        ///
        /// @return Free physical block count.
        uint32_t availableBlocks() const { return budget_.blocks - usedBlocks_; }

        /// @brief Find the longest free contiguous run without changing state.
        ///
        /// @details
        /// The scheduler uses this count before selecting its next allocation.
        ///
        /// @return Largest allocatable run, zero when every block is leased.
        uint32_t largestFreeRun() const;

        /// @brief Reserve a known-fitting consecutive block run.
        ///
        /// @details
        /// First-fit placement affects storage only. The generation validates
        /// that a later release refers to this exact live allocation.
        ///
        /// @param count Positive run size already admitted by the scheduler.
        /// @return A live lease owning exactly count blocks.
        /// @invariant A run of the requested size exists before this call.
        Phase2GpuLease reserveBlocks(uint32_t count);

        /// @brief Resolve a live lease to its device byte address.
        ///
        /// @details
        /// Validates ownership across the whole run; the address stays stable
        /// until release and never participates in semantic comparison.
        ///
        /// @param lease Live lease from this pool.
        /// @return Device address of the lease's first byte.
        /// @invariant Every block in the run has the lease's generation.
        void* address(const Phase2GpuLease& lease) const;

        /// @brief Return a lease after all device work using it has completed.
        ///
        /// @details
        /// Synchronizes before recycling storage so outstanding transfers or
        /// kernels cannot write into the next owner's allocation.
        ///
        /// @param lease Live lease whose ownership is ending.
        /// @return Nothing.
        /// @invariant A released generation cannot be resolved or released twice.
        void release(const Phase2GpuLease& lease);
    };

    /// @brief Suballocate aligned device arrays from one live block lease.
    ///
    /// @details
    /// Buffer objects borrow this arena and must be destroyed before it. The
    /// arena returns its complete lease together, allowing later batches to
    /// redistribute blocks without individual device allocations.
    class CudaPhase2Arena {
        CudaPhase2BlockPool& pool_;
        Phase2GpuLease lease_;
        uint64_t used_{ 0 };

    public:
        /// @brief Acquire a planned block run for one allocation arena.
        ///
        /// @details
        /// The scheduler has already admitted this contiguous block count.
        ///
        /// @param pool Owning GPU reservation, which outlives this arena.
        /// @param blocks Positive number of admitted blocks.
        /// @return An empty arena with a live lease.
        /// @invariant The arena owns its lease until destruction.
        CudaPhase2Arena(CudaPhase2BlockPool& pool, uint32_t blocks)
            : pool_(pool), lease_(pool.reserveBlocks(blocks)) {}

        /// @brief Return every arena block after all borrowers are destroyed.
        ///
        /// @details
        /// Pool release synchronizes outstanding device work before reuse.
        ///
        /// @return Nothing.
        /// @invariant No borrowed array outlives this arena.
        ~CudaPhase2Arena() { pool_.release(lease_); }
        CudaPhase2Arena(const CudaPhase2Arena&) = delete;
        CudaPhase2Arena& operator=(const CudaPhase2Arena&) = delete;

        /// @brief Reserve a 256-byte-aligned array inside the planned arena.
        ///
        /// @details
        /// Array addresses remain stable until the arena is destroyed. An
        /// underestimated layout is a planner bug, asserted before any write.
        ///
        /// @param destination Output device pointer for the borrowed array.
        /// @param bytes Positive byte extent admitted by the layout plan.
        /// @return Nothing.
        /// @invariant The complete array and alignment padding fit this lease.
        void allocate(void** destination, std::size_t bytes) {
            assert(destination != nullptr && bytes > 0);
            const uint64_t aligned = (used_ + 255) & ~uint64_t{255};
            assert(aligned <= capacityBytes());
            assert(bytes <= capacityBytes() - aligned);
            *destination = static_cast<char*>(pool_.address(lease_)) + aligned;
            used_ = aligned + bytes;
        }

        /// @brief Read the arena's allocated byte capacity.
        ///
        /// @details
        /// Includes unused tail bytes that remain available to this arena.
        ///
        /// @return Physical byte extent of the live lease.
        uint64_t capacityBytes() const { return lease_.count * kPhase2GpuBlockBytes; }

        /// @brief Read occupied array bytes including alignment padding.
        ///
        /// @details
        /// This count belongs to allocation accounting, never proof state.
        ///
        /// @return Bytes reserved by calls to allocate.
        uint64_t usedBytes() const { return used_; }
    };

    /// @brief Immutable CUDA device properties required by the Phase 2 backend.
    ///
    /// @details
    /// This pointer-free record is the host-visible result of the startup device
    /// contract. It records only properties that govern kernel compatibility,
    /// fixed allocation sizing, and launch geometry. No CUDA runtime type crosses
    /// the module boundary.
    struct CudaDeviceContract {
        int32_t deviceOrdinal{ 0 };
        int32_t computeMajor{ 0 };
        int32_t computeMinor{ 0 };
        uint64_t totalGlobalBytes{ 0 };
        int32_t multiprocessorCount{ 0 };
        int32_t maximumThreadsPerBlock{ 0 };
    };

    /// @brief Query and validate the active CUDA device for the Phase 2 backend.
    ///
    /// @details
    /// Enumerates CUDA devices, selects the runtime's active device, reads its
    /// immutable properties, and asserts the minimum compute capability required
    /// by the native `sm_89` build. Missing devices, runtime failures, unsupported
    /// capability, or nonsensical sizing are contract violations and never fall
    /// back to the processor backend.
    ///
    /// @return Validated pointer-free properties for the active CUDA device.
    /// @invariant Successful return means at least compute capability 8.9 and
    ///            strictly positive memory, multiprocessor, and block limits.
    /// @see launchCudaContractProbe
    CudaDeviceContract queryCudaDeviceContract();

    /// @brief Launch a real native CUDA kernel and return its deterministic word.
    ///
    /// @details
    /// Allocates one device word, launches the module's native contract kernel,
    /// synchronizes, copies the word back, and frees the device allocation. Every
    /// CUDA operation asserts success. This is the executable build/runtime seam
    /// used by the unit-test gate before larger Phase 2 arenas are introduced.
    ///
    /// @param input Deterministic input word transformed by the device kernel.
    /// @return `input * 1664525 + 1013904223` with unsigned 32-bit wraparound.
    /// @invariant A return proves that native device code launched, synchronized,
    ///            and transferred data successfully in the current process.
    /// @see queryCudaDeviceContract
    uint32_t launchCudaContractProbe(uint32_t input);

    /// @brief Reusable CUDA-event timer for the exact Phase 2 device timeline.
    ///
    /// @details
    /// Owns two opaque CUDA events without exposing CUDA runtime types to host
    /// translation units. One process-owned instance brackets the projection and
    /// task uploads plus every semantic kernel through doom-prefix selection.
    /// Event creation, recording, synchronization, elapsed-time calculation, and
    /// destruction assert on failure; timing never falls back to a host clock.
    class CudaPhase2DeviceTimer final {
    public:
        /// @brief Create both reusable CUDA timing events.
        ///
        /// @details
        /// Allocates the runtime event pair once. The events retain timing support
        /// and are reused by every selected CUDA Phase 2 pass in the process.
        ///
        /// @invariant Successful construction owns two distinct valid CUDA events.
        CudaPhase2DeviceTimer();

        /// @brief Destroy both reusable CUDA timing events.
        ///
        /// @details
        /// Releases the exact pair allocated by the constructor. A runtime failure
        /// asserts because silently leaking a device timing resource violates the
        /// process-owned lifecycle contract.
        ///
        /// @invariant No timing event remains owned after destruction completes.
        ~CudaPhase2DeviceTimer();

        CudaPhase2DeviceTimer(const CudaPhase2DeviceTimer&) = delete;
        CudaPhase2DeviceTimer& operator=(
            const CudaPhase2DeviceTimer&) = delete;

        /// @brief Record the beginning of one exact device timeline interval.
        ///
        /// @details
        /// Enqueues the start event on CUDA's default stream. The caller then
        /// submits the uploads and kernels whose device elapsed time is required.
        ///
        /// @return Nothing.
        /// @invariant A matching `stopSeconds` follows before the next `start`.
        void start();

        /// @brief Finish and return the current device timeline interval.
        ///
        /// @details
        /// Records the finish event on the default stream, synchronizes that exact
        /// event, and returns CUDA's elapsed time between the retained pair. Host
        /// projection and sealing outside the two records are excluded.
        ///
        /// @return Non-negative CUDA event time in seconds.
        /// @invariant `start` has recorded the matching beginning event.
        double stopSeconds();

    private:
        void* startedEvent_{ nullptr };
        void* finishedEvent_{ nullptr };
        bool running_{ false };
    };

    /// @brief Resident lookup operation exercised by the CUDA semantic probe.
    enum class DeviceLookupProbeKind : uint32_t {
        name = 0,
        byteMap = 1,
        reverseMap = 2,
        podMap = 3
    };

    /// @brief Pointer-free input for one direct resident-table CUDA lookup.
    ///
    /// @details
    /// The probe names one uploaded logical block and one lookup family. Byte-key
    /// families consume `key[0..keyLength)`; plain-data maps consume `podKey`.
    /// `mapKind` is ignored for name and reverse probes and otherwise carries the
    /// corresponding `DeviceByteMapKind` or `DevicePodMapKind` value. The inline
    /// byte storage belongs only to the direct lookup contract test; production
    /// request kernels call the same device lookup primitives with task-arena
    /// spans and are not limited by this diagnostic envelope.
    struct DeviceLookupProbe {
        static constexpr uint32_t kMaximumKeyBytes = 128;
        DeviceLookupProbeKind kind{ DeviceLookupProbeKind::name };
        uint32_t logicalBlockIndex{ 0 };
        uint32_t mapKind{ 0 };
        uint32_t keyLength{ 0 };
        int64_t podKey{ 0 };
        char key[kMaximumKeyBytes]{};
    };

    /// @brief Pointer-free result of one resident-table CUDA lookup.
    ///
    /// @details
    /// `recordIndex == -1` is the defined miss result. On a hit, byte and reverse
    /// maps return their arena-global entry index; name probes return the owning
    /// logical block's local `NameId`; plain-data probes return their arena-global
    /// entry index. Payload offsets and counts select blob records, reverse owners,
    /// or plain-data run values. `firstIntValue` and `scalar` expose the first
    /// family-specific payload and scalar metadata for direct semantic comparison.
    struct DeviceLookupProbeResult {
        int32_t recordIndex{ -1 };
        uint32_t payloadOffset{ 0 };
        uint32_t payloadCount{ 0 };
        int32_t firstIntValue{ 0 };
        uint64_t scalar{ 0 };
    };

    /// @brief Pointer-free input for one normalized-key and owner-gate CUDA twin.
    ///
    /// @details
    /// Names a resident logical block, one of its four hash-memory registries, and
    /// an ordered candidate path in logical-block-local statement indices. The
    /// diagnostic kernel applies the exact normalized-key build plus whole-key and
    /// owner-subkey probes that production frontier kernels reuse.
    struct DeviceGrowthCandidateProbe {
        uint32_t logicalBlockIndex{ 0 };
        DeviceHashMemoryKind memory{ DeviceHashMemoryKind::overall };
        NameId statementIndices[ExecutionParameters::MAX_EXPRESSIONS]{};
        NameId count{ 0 };
    };

    /// @brief Host-visible result of one device candidate-key semantic probe.
    ///
    /// @details
    /// Carries the exact normalized-key payload and separates subkey presence,
    /// owner-signature satisfaction, and whole-key presence. Separating the flags
    /// proves the three contracts independently, including a present subkey whose
    /// u_ owner signatures reject the candidate.
    struct DeviceGrowthCandidateProbeResult {
        NameId normalizedKey[ExecutionParameters::MAX_KEY_SLOTS]{};
        NameId normalizedKeyLength{ 0 };
        uint32_t subkeyPresent{ 0 };
        uint32_t subkeySatisfied{ 0 };
        uint32_t wholeKeyPresent{ 0 };
    };

    /// @brief Immutable request-shape limits consumed by device frontier growth.
    ///
    /// @details
    /// Carries the three analyzer parameters that are not resident logical-block
    /// state plus the measured remaining-span cutoff that selects cooperative
    /// candidate distribution. Census mode zero is production, mode one records
    /// span shape, and mode two selects separate gate-census kernel specializations
    /// as well. This retains exact test coverage without charging production
    /// kernels for per-thread observation state.
    /// The selected and overall registry key lengths remain in each
    /// `DeviceLogicalBlockProjection` and are therefore not duplicated here.
    struct DevicePhase2GrowthParameters {
        int32_t maximumHypothesisKeyLength{ 0 };
        int32_t maximumSecondaryVariables{ 0 };
        int32_t maximumSecondaryVariablesOrint{ 0 };
        uint32_t cooperativeSpanThreshold{ 0 };
        uint32_t collectSpanCensus{ 0 };
    };

    /// Number of exact remaining-candidate span buckets reported by GPU growth.
    inline constexpr uint32_t kDeviceGrowthSpanBucketCount = 12;

    /// Candidate depths zero through the fixed maximum expression count.
    inline constexpr uint32_t kDeviceGrowthCensusDepthCount =
        static_cast<uint32_t>(ExecutionParameters::MAX_EXPRESSIONS) + 1;

    /// First census-selected production cutoff for cooperative growth candidates.
    inline constexpr uint32_t kDeviceGrowthProductionCooperativeSpan = 64;

    /// @brief Observation-only cumulative growth-gate counts at one depth.
    ///
    /// @details
    /// Counts every attempted appended premise, then the candidates remaining
    /// after mandatory reachability, validity comparability, hypothesis shape,
    /// secondary-variable shape, and key-length gates. Map and event fields are
    /// branch outcomes from the surviving key probes rather than a single
    /// cumulative chain. No semantic kernel reads these values.
    struct Phase2GrowthGateCensus {
        uint64_t candidateAttempts{ 0 };
        uint64_t mandatoryReachable{ 0 };
        uint64_t validityComparable{ 0 };
        uint64_t hypothesisCompatible{ 0 };
        uint64_t secondaryCompatible{ 0 };
        uint64_t keyLengthAllowed{ 0 };
        uint64_t subkeyPresent{ 0 };
        uint64_t ownerSatisfied{ 0 };
        uint64_t wholeKeyPresent{ 0 };
        uint64_t termsSatisfied{ 0 };
        uint64_t acceptedEvents{ 0 };
        uint64_t children{ 0 };
    };

    /// @brief Host-visible bounded result of one bulk request-growth launch.
    ///
    /// @details
    /// Reports the persistent semantic ledgers, the largest ping-pong frontier,
    /// and an observation-only census of the live node work shape and cumulative
    /// candidate gates at each appended-premise depth. Span bucket zero contains
    /// no remaining positions, bucket one contains one, buckets two through ten
    /// contain `2^(b-1)` through `2^b-1`, and bucket eleven contains 1,024 or
    /// more. Event array order is deliberately unspecified until the later exact
    /// processor-order reconstruction pass.
    /// The three prefix maxima count `NameId` values simultaneously required by
    /// one live frontier: normalized payload, distinct normalization variables,
    /// and distinct secondary-variable identifiers. They size the approved
    /// fixed pooled-prefix representation without influencing proof flow.
    struct Phase2GrowthResult {
        uint32_t acceptedEventCount{ 0 };
        uint32_t rawRequestCount{ 0 };
        uint32_t maximumFrontierCount{ 0 };
        uint32_t maximumCooperativeNodeCount{ 0 };
        uint64_t maximumPrefixPayloadValues{ 0 };
        uint64_t maximumPrefixVariableValues{ 0 };
        uint64_t maximumPrefixSecondaryValues{ 0 };
        uint64_t spanNodeCounts[kDeviceGrowthSpanBucketCount]{};
        uint64_t spanCandidateCounts[kDeviceGrowthSpanBucketCount]{};
        Phase2GrowthGateCensus gateDepthCounts[
            kDeviceGrowthCensusDepthCount]{};
    };

    /// @brief Complete logical growth totals independent of processing windows.
    struct Phase2GrowthStreamResult {
        uint64_t acceptedEventCount{ 0 };
        uint64_t rawRequestCount{ 0 };
        uint64_t maximumFrontierCount{ 0 };
        uint64_t maximumHostBackingBytes{ 0 };
        uint64_t uploadedFrontierBytes{ 0 };
        uint64_t frontierTransferCalls{0};
        uint32_t frontierTransferRows{0};
        uint64_t downloadedFrontierBytes{ 0 };
        uint64_t drainCount{ 0 };
        uint64_t cutoffDrainCount{0}, prunedFrontierNodes{0};
        uint64_t maximumPrefixPayloadValues{0}, maximumPrefixVariableValues{0};
        uint64_t maximumPrefixSecondaryValues{0}, maximumCandidateWindow{0};
        uint64_t prefixWindows{0}, candidateWindows{0};
        uint64_t candidateNodeWindowQueries{0};
        uint64_t unslicedGateNodes{0}, scheduledGateNodes{0};
        uint64_t candidatePositionsBeforePrefilter{0}, candidatePositionsAfterPrefilter{0};
        uint64_t frontierSpills{0};
        uint64_t probeWindows{0}, exactCountReads{0};
    };

    /// @brief Bounded host rows for request-identity transfer pages.
    inline constexpr uint32_t phase2FrontierTransferRows = 1024;

    /// @brief Preferred pinned frontier-copy rows, bounded by one host-pool block.
    inline constexpr uint32_t phase2FrontierCopyRows = 4096;

    /// @brief Maximum consecutive frontier rows sharing one GPU preparation.
    /// @details Host transfers remain paged; actual device and prefix capacities
    ///          may split this execution window without changing the search.
    inline constexpr uint32_t phase2FrontierExecutionRows = 65536;

    /// @brief Largest planned node count of one pass-local frontier-store buffer.
    ///
    /// @details 4,194,304 nodes of 56 bytes (224 MiB per buffer). The planner
    /// sizes the store from the measured level demand up to this ceiling; a
    /// larger level spills its oldest rows to host pages.
    inline constexpr uint32_t phase2FrontierStoreMaximumRows = uint32_t{1} << 22;

    /// @brief Remove the frontier nodes of cut-off growth calls in place,
    ///        keeping the survivors in their order.
    ///
    /// @details
    /// The host half of the level-end cutoff in `runRequestGrowthWindows`. One
    /// stable forward compaction walks the frontier page span by page span
    /// (`PagedVector::contiguousRun`), reads each node's decision from the
    /// per-call keep flags, writes a survivor only when an earlier node was
    /// removed, and releases the tail with one `truncate`. The survivors and
    /// their order equal the node-by-node predicate walk over the same flags.
    ///
    /// @param frontier   The completed level's frontier, compacted in place.
    /// @param keepByCall One flag per scheduled growth call; 0 cuts the call's
    ///                   nodes off, any other value keeps them.
    /// @param callCount  Number of flags; every node's `callIndex` is below it.
    /// @return The number of removed nodes.
    /// @invariant Surviving nodes keep their relative order.
    /// @see CudaPhase2GrowthBuffer::runRequestGrowthWindows.
    uint64_t pruneFrontierByCall(PagedVector<DeviceGrowthNode>& frontier,
        const uint8_t* keepByCall, uint32_t callCount);

    /// @brief Remove the device-resident frontier rows of cut-off growth calls,
    ///        keeping the survivors in their order.
    ///
    /// @details
    /// The device half of the level-end cutoff in `runRequestGrowthWindows`: the
    /// rows of the current level that stay in the pass-local frontier store are
    /// compacted by one stable `cub::DeviceSelect::If` into the other store
    /// buffer, judged by the per-call keep flags already on the device. The
    /// survivors and their order equal `pruneFrontierByCall` over the same rows
    /// and flags. Synchronous: the survivor count is read back before return.
    ///
    /// @param deviceRows       The level's device rows.
    /// @param count            Number of rows; may be 0.
    /// @param deviceKeepByCall One device flag per scheduled growth call; 0 cuts
    ///                         the call's rows off.
    /// @param deviceOutput     Destination buffer (other store buffer), at least
    ///                         @p count rows, disjoint from @p deviceRows.
    /// @param deviceTemporary  CUB workspace.
    /// @param temporaryBytes   Its size, at least the selection's requirement.
    /// @param deviceSelected   One device word receiving the survivor count.
    /// @return The number of surviving rows.
    /// @invariant Surviving rows keep their relative order.
    /// @see pruneFrontierByCall, CudaPhase2GrowthBuffer::runRequestGrowthWindows.
    uint32_t pruneDeviceFrontierByCall(const DeviceGrowthNode* deviceRows,
        uint32_t count, const uint8_t* deviceKeepByCall, DeviceGrowthNode* deviceOutput,
        void* deviceTemporary, std::size_t temporaryBytes, uint32_t* deviceSelected);

    /// @brief Copy device words into host-mapped memory on the compute engine.
    ///
    /// @details
    /// Launches one 64-thread block on the default stream, behind every launch
    /// it observes, that copies @p words 32-bit words from @p source into the
    /// device view of @p mappedDestination and fences them system-wide. A
    /// small read-back therefore needs no copy-engine transfer, which on the
    /// Windows display driver model waits for an engine switch on both sides
    /// (tens of microseconds each). Asynchronous: the words are visible to
    /// the host once an event recorded after this call has completed.
    ///
    /// @param source            Device words, 4-byte aligned.
    /// @param words             Number of words; positive.
    /// @param mappedDestination Host memory pinned by `cudaHostAlloc` (mapped)
    ///                          or `cudaHostRegister`, 4-byte aligned, at least
    ///                          @p words words; may point inside the pinned range.
    /// @return Nothing.
    /// @invariant The destination is complete once a later event on the default
    ///            stream has completed.
    void publishDeviceWords(const void* source, uint32_t words, void* mappedDestination);

    /// @brief Run `pruneDeviceFrontierByCall` on host rows through private
    ///        device copies.
    ///
    /// @details
    /// The unit-test seam of the device cutoff: uploads @p hostRows and
    /// @p hostKeepByCall, runs the production selection, downloads the
    /// survivors into @p hostOutput and frees its device copies. Zero rows is
    /// a defined no-op returning 0.
    ///
    /// @param hostRows       Host frontier rows.
    /// @param count          Number of rows.
    /// @param hostKeepByCall One host flag per growth call.
    /// @param callCount      Number of flags; every row's `callIndex` is below it.
    /// @param hostOutput     Host destination for the survivors, @p count rows.
    /// @return The number of surviving rows.
    /// @see pruneDeviceFrontierByCall.
    uint32_t pruneDeviceFrontierByCallOnHost(const DeviceGrowthNode* hostRows,
        uint32_t count, const uint8_t* hostKeepByCall, uint32_t callCount,
        DeviceGrowthNode* hostOutput);

    /// @brief The three measured prefix sizes of one frontier node, as the host
    ///        schedules prefix windows from them.
    struct Phase2PrefixSizes {
        uint16_t payloadLength{0};
        uint16_t variableCount{0};
        uint16_t secondaryCount{0};
    };

    /// @brief Unpack one node's prefix sizes from the word the device packed.
    ///
    /// @details
    /// Bits 0-15 hold the payload length, 16-31 the variable count and 32-47 the
    /// secondary count, as `packPhase2PrefixSizes` writes them; bits 48-63 are
    /// zero. The host needs only these three fields to pack consecutive prefixes
    /// into execution windows, so a frontier page downloads 8 bytes per node
    /// instead of the whole `DeviceGrowthPrefix` record.
    ///
    /// @param word One packed word.
    /// @return The three sizes.
    inline Phase2PrefixSizes unpackPhase2PrefixSizes(uint64_t word) {
        assert((word >> 48) == 0);
        return Phase2PrefixSizes{static_cast<uint16_t>(word & 0xffffu),
            static_cast<uint16_t>((word >> 16) & 0xffffu),
            static_cast<uint16_t>((word >> 32) & 0xffffu)};
    }

    /// @brief Pack the measured prefix sizes of `count` device prefix records
    ///        into one 64-bit word each, on the device.
    ///
    /// @details
    /// Launches one thread per record on the default stream; the words keep the
    /// record order. The layout is the one `unpackPhase2PrefixSizes` reads. The
    /// call returns after the launch; stream order places a following copy
    /// behind it.
    ///
    /// @param devicePrefixes Device prefix records, as the measuring
    ///        preparation wrote them.
    /// @param count Number of records; 0 launches nothing.
    /// @param devicePacked Device output, one word per record.
    /// @return Nothing.
    void packPhase2PrefixSizes(const DeviceGrowthPrefix* devicePrefixes,
        uint32_t count, uint64_t* devicePacked);

    /// @brief Pack prefix sizes through the device for host-resident records:
    ///        upload, `packPhase2PrefixSizes`, download.
    ///
    /// @details
    /// The unit-test seam of the growth driver's prefix-size transfer: runs the
    /// production packing kernel on a private device copy of @p hostPrefixes and
    /// returns the packed words, synchronously. Zero records is a defined no-op.
    ///
    /// @param hostPrefixes Host prefix records.
    /// @param count Number of records.
    /// @param hostPacked Host output, one packed word per record.
    /// @return Nothing.
    void packPhase2PrefixSizesOnHost(const DeviceGrowthPrefix* hostPrefixes,
        uint32_t count, uint64_t* hostPacked);

    /// @brief Fixed ceilings for exact event ordering and request tokens.
    ///
    /// @details
    /// Event capacity matches the persistent growth ledger and request capacity matches
    /// the retained raw-request ceiling.
    struct Phase2OrderingCapacity {
        uint32_t events{ 0 };
        uint32_t requests{ 0 };
    };

    /// @brief Audited ordering capacity for the FTA shortcut route.
    inline constexpr Phase2OrderingCapacity
        kFtaShortcutPhase2OrderingCapacity{
            4194304, 262144 };

    /// @brief Ordering capacity for all full-run routes, matching the growth event ledger.
    inline constexpr Phase2OrderingCapacity kFullRunPhase2OrderingCapacity{
        6291456, 524288 };

    /// @brief One exact-order request token for later device evaluation.
    ///
    /// @details
    /// References the persistent growth event instead of copying its path.
    /// `eventOrder` is the global task/call/run/processor-walk position;
    /// `growthPosition` is the inclusive accepted-subkey tally within that task.
    struct DeviceOrderedRequestToken {
        uint64_t growthPosition{ 0 };
        uint32_t eventIndex{ 0 };
        uint32_t eventOrder{ 0 };
    };

    /// @brief Host-visible counts after exact event ordering.
    struct Phase2OrderingResult {
        uint32_t orderedEventCount{ 0 };
        uint32_t requestCount{ 0 };
    };


    /// @brief Wide logical expansion totals and physical evaluation occupancy.
    struct Phase2EvaluationStreamMetrics {
        uint64_t owners{0}, candidates{0}, hits{0}, values{0};
        uint64_t ownerWindows{0}, valueWindows{0};
        uint32_t maximumOwnerWindow{0}, maximumValueWindow{0};
    };

    /// @brief Fixed ceilings for complete device request evaluation.
    ///
    /// @details
    /// The first field bounds per-logical-block doom output, the next six bound
    /// request-to-value work expansion, and the remaining fields bound complete
    /// provenance-bearing firing output. Every
    /// value is a retained FTA next-power-of-two ceiling and every paired device
    /// array is allocated once by `CudaPhase2EvaluationBuffer`.
    struct Phase2EvaluationCapacity {
        uint32_t logicalBlocks{ 0 };
        uint32_t requests{ 0 };
        uint32_t reverseOwners{ 0 };
        uint32_t candidateOwners{ 0 };
        uint32_t encodedHits{ 0 };
        uint32_t localValues{ 0 };
        uint32_t firingRecords{ 0 };
        uint32_t generatedBytes{ 0 };
        uint32_t levelValues{ 0 };
        uint32_t originDependencies{ 0 };
        uint32_t markerKeys{ 0 };
        uint32_t markerRemainingArgs{ 0 };
        uint32_t markerArgs{ 0 };
    };

    /// @brief Audited evaluation capacity for the FTA shortcut route.
    inline constexpr Phase2EvaluationCapacity
        kFtaShortcutPhase2EvaluationCapacity{
            kMaxProjectedBlocksPerChunk,
            262144,
            4194304,
            2097152,
            2097152,
            4194304,
            2097152,
            134217728,
            16777216,
            16777216,
            131072,
            2097152,
            2097152 };

    /// @brief Common evaluation capacity for full execution and shortcut replay.
    inline constexpr Phase2EvaluationCapacity kFullRunPhase2EvaluationCapacity{
        kMaxProjectedBlocksPerChunk,
        524288,
        4194304,
        524288,
        524288,
        1048576,
        524288,
        33554432,
        2097152,
        2097152,
        524288,
        262144,
        524288 };

    /// @brief One generated-byte or generated-substring view in evaluator output.
    struct DeviceEvaluationByteSlice {
        uint32_t offset{ 0 };
        uint32_t length{ 0 };
    };

    /// @brief One identifier-backed provenance dependency.
    struct DeviceEvaluationDependency {
        NameId originalId{ 0 };
        NameId validityId{ 0 };
    };

    /// @brief Per-request state after dependency, validity, and reverse lookup.
    struct DeviceEvaluationRequestState {
        uint32_t logicalBlockIndex{ 0 };
        int32_t reverseEntryIndex{ -1 };
        uint32_t reverseOwnerCount{ 0 };
        NameId consensusValidityId{ 0 };
        int32_t maximumPremiseIteration{ -1 };
        uint32_t pure{ 0 };
        uint32_t active{ 0 };
    };

    /// @brief One request-to-remaining-argument owner expansion item.
    struct DeviceEvaluationOwnerWork {
        uint32_t requestIndex{ 0 };
        uint32_t remainingEntryIndex{ 0 };
    };

    /// @brief One subset-surviving remaining-argument owner and encoded lookup.
    struct DeviceEvaluationCandidate {
        uint32_t requestIndex{ 0 };
        uint32_t remainingEntryIndex{ 0 };
        int32_t encodedEntryIndex{ -1 };
    };

    /// @brief One LocalMemoryValue blob selected for semantic evaluation.
    struct DeviceEvaluationValueWork {
        uint32_t requestIndex{ 0 };
        uint32_t blobRecordIndex{ 0 };
        uint32_t remainingEntryIndex{ 0 };
    };

    /// @brief Stable bit vocabulary for device firing-record classification.
    ///
    /// @details
    /// The low six bits preserve the source `LocalMemoryValue` classification
    /// exactly. The remaining bits are the context-derived processor verdicts
    /// filled by later evaluator stages. Keeping both halves in one word lets the
    /// final canonical handoff consume GPU output without reopening the LMV blob.
    // Serialized RuleJustification value; the host sealing boundary checks it.
    inline constexpr int32_t deviceIntegrationJustification = 2;

    enum DevicePhase2FiringFlag : uint32_t {
        deviceFiringMarker = 1u << 0,
        deviceFiringOrdis2Demand = 1u << 1,
        deviceFiringProductOfDisintegration = 1u << 2,
        deviceFiringDisintegrationAllowed = 1u << 3,
        deviceFiringOrdisOnly = 1u << 4,
        deviceFiringSubsetExclusion = 1u << 5,
        deviceFiringDoNotDisintegrate = 1u << 6,
        deviceFiringAllowOrDisintegration = 1u << 7,
        // Bit 8 is retired (transitive mail permission).
        deviceFiringAlreadyKnown = 1u << 9,
        // Bit 10 is retired (marker mail permission).
        deviceFiringRecursionHead = 1u << 11,
        deviceFiringDoomTrigger = 1u << 12
    };

    /// @brief Complete fixed-header representation of one device firing output.
    ///
    /// @details
    /// Variable payloads address the evaluator's byte, integer, dependency, and
    /// marker-reference arenas. The request and LMV references keep all source
    /// implication and premise identifiers available for processor-side sealing
    /// without semantic replay. Logical-block, part, and growth-position fields
    /// retain the exact executor/request identity needed by canonical grouping
    /// and the later doom-prefix reduction.
    struct DevicePhase2FiringRecord {
        uint64_t growthPosition{ 0 };
        uint32_t requestIndex{ 0 };
        uint32_t logicalBlockIndex{ 0 };
        uint32_t partOrdinal{ 0 };
        uint32_t blobRecordIndex{ 0 };
        DeviceEvaluationByteSlice expression{};
        uint32_t levelsOffset{ 0 };
        uint32_t levelsCount{ 0 };
        uint32_t originDependencyOffset{ 0 };
        uint32_t originDependencyCount{ 0 };
        uint32_t markerKeyOffset{ 0 };
        uint32_t markerKeyCount{ 0 };
        uint32_t markerRemainingArgOffset{ 0 };
        uint32_t markerRemainingArgCount{ 0 };
        uint32_t markerArgOffset{ 0 };
        uint32_t markerArgCount{ 0 };
        NameId validityId{ 0 };
        int32_t iteration{ -1 };
        int32_t demandSourceImplId{ 0 };
        int32_t standardMaxAdmissionDepth{ 0 };
        int32_t standardMaxSecondaryNumber{ 0 };
        uint32_t flags{ 0 };
    };

    /// @brief Host-visible counts after device evaluation work expansion.
    struct Phase2EvaluationResult {
        uint32_t requestCount{ 0 };
        uint32_t dependencyPassCount{ 0 };
        uint32_t reverseOwnerCount{ 0 };
        uint32_t candidateOwnerCount{ 0 };
        uint32_t encodedHitCount{ 0 };
        uint32_t localValueCount{ 0 };
    };

    /// @brief Host-visible counts after GPU firing-expression materialization.
    ///
    /// @details
    /// Counts only LMVs that survive rule-scope, closed-scope, marker-purity, and
    /// ordis2 depth gates. `generatedByteCount` is the exact used prefix of the
    /// fixed generated-byte arena, including scratch emitted before late gates.
    struct Phase2FiringExpressionResult {
        uint32_t firingRecordCount{ 0 };
        uint32_t generatedByteCount{ 0 };
        uint32_t levelValueCount{ 0 };
        uint32_t originDependencyCount{ 0 };
        uint32_t markerKeyCount{ 0 };
        uint32_t markerRemainingArgCount{ 0 };
        uint32_t markerArgCount{ 0 };
    };

    /// @brief Next value and marker-key positions in a prepared value window.
    struct Phase2FiringCursor {
        uint32_t value{ 0 };
        uint32_t markerKey{ 0 };
    };

    /// @brief Defined outcomes of a bounded firing-output scheduling step.
    enum class Phase2FiringWindowState { complete, ready, needsCapacity };

    /// @brief One bounded output download or a retained capacity request.
    ///
    /// @details
    /// A ready marker fragment names its position in the complete marker-key
    /// span. A needs-capacity result leaves its cursor unchanged and reports the
    /// next fragment's simultaneous column requirements.
    struct Phase2FiringWindow {
        Phase2FiringWindowState state{ Phase2FiringWindowState::complete };
        Phase2FiringExpressionResult used{};
        Phase2OutputUsage required{};
        uint32_t valueBegin{ 0 };
        uint32_t valueCount{ 0 };
        uint32_t markerKeyBegin{ 0 };
        uint32_t markerKeyCount{ 0 };
        uint32_t totalMarkerKeys{ 0 };
        bool markerFragment{ false };
    };

    static_assert(std::is_trivially_copyable_v<DeviceLookupProbe>);
    static_assert(std::is_trivially_copyable_v<DeviceLookupProbeResult>);
    static_assert(std::is_trivially_copyable_v<DeviceGrowthCandidateProbe>);
    static_assert(std::is_trivially_copyable_v<
        DeviceGrowthCandidateProbeResult>);
    static_assert(std::is_trivially_copyable_v<
        DevicePhase2GrowthParameters>);
    static_assert(std::is_trivially_copyable_v<Phase2GrowthGateCensus>);
    static_assert(std::is_trivially_copyable_v<Phase2GrowthResult>);
    static_assert(std::is_trivially_copyable_v<Phase2OrderingCapacity>);
    static_assert(std::is_trivially_copyable_v<DeviceOrderedRequestToken>);
    static_assert(std::is_trivially_copyable_v<Phase2OrderingResult>);
    static_assert(std::is_trivially_copyable_v<Phase2EvaluationCapacity>);
    static_assert(std::is_trivially_copyable_v<DeviceEvaluationByteSlice>);
    static_assert(std::is_trivially_copyable_v<DeviceEvaluationDependency>);
    static_assert(std::is_trivially_copyable_v<DeviceEvaluationRequestState>);
    static_assert(std::is_trivially_copyable_v<DeviceEvaluationOwnerWork>);
    static_assert(std::is_trivially_copyable_v<DeviceEvaluationCandidate>);
    static_assert(std::is_trivially_copyable_v<DeviceEvaluationValueWork>);
    static_assert(std::is_trivially_copyable_v<DevicePhase2FiringRecord>);
    static_assert(std::is_trivially_copyable_v<Phase2EvaluationResult>);
    static_assert(std::is_trivially_copyable_v<Phase2FiringExpressionResult>);
    static_assert(sizeof(DeviceLookupProbe) == 152);
    static_assert(sizeof(DeviceLookupProbeResult) == 24);
    static_assert(sizeof(DeviceGrowthCandidateProbe) == 44);
    static_assert(sizeof(DeviceGrowthCandidateProbeResult) == 1040);
    static_assert(sizeof(DevicePhase2GrowthParameters) == 20);
    static_assert(sizeof(Phase2GrowthGateCensus) == 96);
    static_assert(sizeof(Phase2GrowthResult) == 1096);
    static_assert(sizeof(Phase2OrderingCapacity) == 8);
    static_assert(sizeof(DeviceOrderedRequestToken) == 16);
    static_assert(sizeof(Phase2OrderingResult) == 8);
    static_assert(sizeof(Phase2EvaluationCapacity) == 52);
    static_assert(sizeof(DeviceEvaluationByteSlice) == 8);
    static_assert(sizeof(DeviceEvaluationDependency) == 8);
    static_assert(sizeof(DeviceEvaluationRequestState) == 28);
    static_assert(sizeof(DeviceEvaluationOwnerWork) == 8);
    static_assert(sizeof(DeviceEvaluationCandidate) == 12);
    static_assert(sizeof(DeviceEvaluationValueWork) == 12);
    static_assert(sizeof(DevicePhase2FiringRecord) == 96);
    static_assert(sizeof(Phase2EvaluationResult) == 24);
    static_assert(sizeof(Phase2FiringExpressionResult) == 28);

    /// @brief Fixed-capacity CUDA ownership for one reusable Phase 2 projection.
    ///
    /// @details
    /// Construction allocates device arrays for every projection column plus one
    /// reusable result scratch. Uploads copy only used prefixes after asserting
    /// they fit;
    /// no upload allocates, grows, or substitutes host execution. The class owns
    /// opaque host-side device addresses, while every record stored on the device
    /// remains pointer-free and offset-based.
    class CudaPhase2ProjectionBuffer {
    public:
        /// @brief Measure device and aligned arena bytes before creating a buffer.
        ///
        /// @details
        /// Validates the capacity shape and queries current CUDA library workspace
        /// without allocating device arrays. Includes every processing column and
        /// the 256-byte padding required by the common arena.
        ///
        /// @param capacity Proposed processing and input column capacities.
        /// @return Device payload bytes and exact aligned arena footprint.
        /// @invariant No buffer ownership or proof state is created or changed.
        static Phase2DeviceAllocationSize measureAllocation(Phase2ProjectionCapacity capacity);

        /// @brief Allocate all device projection storage once at engine startup.
        ///
        /// @details
        /// Validates the active CUDA device and allocates fixed arrays for logical
        /// blocks plus all 23 subordinate semantic columns and the checksum
        /// result. Every capacity is positive and every CUDA allocation asserts.
        ///
        /// @param fixedCapacity Immutable device element and byte ceilings.
        /// @param allocation Optional shared arena; null owns standalone arrays.
        /// @return An empty device buffer owning all declared allocations.
        /// @invariant No method changes an allocation address or capacity.
        explicit CudaPhase2ProjectionBuffer(
            Phase2ProjectionCapacity fixedCapacity, CudaPhase2Arena* allocation = nullptr);

        /// @brief Release every device allocation owned by this projection buffer.
        ///
        /// @details
        /// Frees each startup allocation exactly once and asserts every CUDA free.
        /// A buffer is neither copyable nor movable, so ownership cannot alias.
        ///
        /// @return Nothing.
        ~CudaPhase2ProjectionBuffer();

        CudaPhase2ProjectionBuffer(const CudaPhase2ProjectionBuffer&) = delete;
        CudaPhase2ProjectionBuffer& operator=(
            const CudaPhase2ProjectionBuffer&) = delete;
        CudaPhase2ProjectionBuffer(CudaPhase2ProjectionBuffer&&) = delete;
        CudaPhase2ProjectionBuffer& operator=(
            CudaPhase2ProjectionBuffer&&) = delete;

        /// @brief Copy one host projection's used prefixes into fixed device arrays.
        ///
        /// @details
        /// Asserts each host used length against the constructor ceiling, then
        /// copies all 24 semantic columns in canonical array order. Empty columns
        /// are defined and issue no zero-byte CUDA call. The method stores used
        /// counts for the next kernel launch.
        ///
        /// @param hostProjection Fixed-capacity host image to upload byte-for-byte.
        /// @return Nothing.
        /// @invariant Device used prefixes equal the supplied host prefixes after
        ///            return; unused suffixes are semantically inaccessible.
        void upload(const Phase2ProjectionArena& hostProjection);

        /// @brief Start a dependency-staged Phase 2 projection upload.
        ///
        /// @details
        /// Copies filter, growth, and ordering columns synchronously, then queues
        /// evaluation-only rule-string, reverse-map, run-value, and metadata
        /// prefixes through process-owned page-locked staging arrays on a
        /// nonblocking CUDA stream. Kernels may consume only the early column set
        /// until `finishPhase2Upload` returns.
        ///
        /// @param hostProjection Fixed-capacity host image to upload byte-for-byte.
        /// @return Nothing.
        /// @invariant No earlier staged upload is pending and every queued prefix
        ///            remains immutable until `finishPhase2Upload`.
        void beginPhase2Upload(
            const Phase2ProjectionArena& hostProjection);

        /// @brief Start a complete device image assembled from host windows.
        ///
        /// @details Resets uploaded lengths without reallocating device storage.
        /// Each window appends synchronously at its measured rebased offsets.
        /// finishPhase2Upload seals the image before any semantic kernel.
        /// @return Nothing.
        /// @invariant No earlier upload is pending.
        void beginStreamedUpload();

        /// @brief Append a rebased complete host window to the device image.
        ///
        /// @details Copies only used ranges into preallocated device columns.
        /// Completion releases the source window's lifetime; no device work is
        /// executed and all complete-image capacities remain unchanged.
        /// @param source Rebased window with stable host columns during this call.
        /// @param bases Assigned offsets, equal to the uploaded ends.
        /// @return Nothing.
        /// @invariant A streamed upload is pending and every append fits.
        void appendStreamedUpload(const Phase2ProjectionArena& source,
            const Phase2ProjectionArena::ShardBases& bases);

        /// @brief Seal a completed deferred or streamed projection upload.
        ///
        /// @details
        /// Synchronizes the process-owned nonblocking stream after independent
        /// host schedule construction has had an opportunity to overlap its
        /// copies. Every projection column is device-resident when the method
        /// returns, before the first semantic kernel starts.
        ///
        /// @return Nothing.
        /// @invariant Exactly one deferred or streamed upload is pending on entry.
        void finishPhase2Upload();

        /// @brief Hash the uploaded projection bytes on the CUDA device.
        ///
        /// @details
        /// Launches one deterministic verification thread over all 24 uploaded
        /// semantic columns in their fixed canonical order, synchronizes, and
        /// returns the copied 64-bit FNV-1a result. This is a projection transfer
        /// twin, not the eventual parallel request kernel.
        ///
        /// @return Device-computed checksum of all uploaded used prefixes.
        /// @invariant Requires at least one uploaded logical-block descriptor.
        uint64_t launchChecksum() const;

        /// @brief Execute one direct CUDA lookup against the uploaded projection.
        ///
        /// @details
        /// Launches a one-thread diagnostic kernel that exercises the exact device
        /// probes used by later parallel request kernels. The operation covers name
        /// interning, all byte-key views, the derived remaining-argument reverse
        /// map, and all plain-data views. A missing key is a defined result with
        /// `recordIndex == -1`; malformed kinds, indices, or lengths assert.
        ///
        /// @param probe Pointer-free lookup family, logical block, and key.
        /// @return Device-computed lookup result copied from fixed result scratch.
        /// @invariant `upload` has supplied the logical block and every selected
        ///            view before this method is called.
        DeviceLookupProbeResult launchLookupProbe(
            const DeviceLookupProbe& probe) const;

        /// @brief Build and probe one complete candidate on the CUDA device.
        ///
        /// @details
        /// Launches the reusable device normalized-key builder, probes the selected
        /// whole-key and subkey tables, and applies the owner-set u_ signature gate
        /// from three premises onward. The method performs no processor replay and
        /// returns the device verdict through fixed result scratch.
        ///
        /// @param probe Logical block, registry, and ordered statement-index path.
        /// @return Device-built key payload and independent lookup verdicts.
        /// @invariant `upload` supplied every named statement and registry; probe
        ///            count is in `[1, MAX_EXPRESSIONS]`.
        DeviceGrowthCandidateProbeResult launchGrowthCandidateProbe(
            const DeviceGrowthCandidateProbe& probe) const;

        /// @brief Report every fixed projection allocation including probe scratch.
        ///
        /// @details
        /// Returns the exact sum of the 24 immutable projection-column allocation
        /// requests and the one reusable growth-probe result allocation captured
        /// during construction. Uploads do not change this ownership total.
        ///
        /// @return Exact startup bytes owned by this projection buffer.
        /// @invariant The value is constant for the object's lifetime.
        uint64_t fixedAllocationBytes() const;

        /// @brief Page-locked host bytes this buffer holds for its deferred
        ///        upload columns (`cudaHostAlloc`).
        ///
        /// @details
        /// The deferred columns of `beginPhase2Upload` are staged through pinned
        /// host memory so the upload overlaps projection work; the first deferred
        /// upload allocates that staging and it stays until destruction. A buffer
        /// filled only by streamed windows never stages and reports zero
        /// (D-447). Pinned pages cannot be paged out, so they are real RAM.
        /// Observability only (Rule 16).
        ///
        /// @return Pinned host bytes.
        uint64_t pinnedHostBytes() const { return pinnedHostBytes_; }

    private:
        bool ownsDeviceAllocations_{ true };
        friend class CudaPhase2FilterSortBuffer;
        friend class CudaPhase2GrowthBuffer;
        friend class CudaPhase2OrderingBuffer;
        friend class CudaPhase2EvaluationBuffer;
        static constexpr uint32_t kProjectionColumnCount = 24;
        inline static constexpr uint32_t kDeferredPhase2Columns[9] = {
            5, 6, 13, 14, 15, 16, 17, 21, 23
        };
        Phase2ProjectionCapacity capacity_{};
        void* deviceColumns_[kProjectionColumnCount]{};
        void* deviceResultScratch_{ nullptr };
        void* deferredUploadStream_{ nullptr };
        void* deferredHostColumns_[kProjectionColumnCount]{};
        uint64_t deferredStagingBytes_[kProjectionColumnCount]{};
        uint32_t usedCounts_[kProjectionColumnCount]{};
        bool streamedUpload_ = false;
        uint64_t fixedAllocationBytes_{ 0 };
        uint64_t pinnedHostBytes_{ 0 };  // cudaHostAlloc total of the deferred columns, once staged
        bool phase2UploadPending_{ false };
    };

    /// @brief Upload thread for the completed construction windows of one
    ///        streamed projection.
    ///
    /// @details
    /// The CUDA projection builds a chunk's host image in consecutive
    /// construction windows (D-436). This thread copies each completed
    /// window into the device image (`CudaPhase2ProjectionBuffer::appendStreamedUpload`)
    /// and appends its retained sealing data
    /// (`Phase2ProjectionArena::appendSealingProjection`) while the caller builds
    /// the next window, so at most two windows are alive: one uploading and one
    /// under construction. Windows run strictly in submission order, so device
    /// offsets and the retained sealing image equal those of the synchronous
    /// sequence. A finished window is released on the submitting thread, which
    /// allocated it, at the next `submit` or at `finish`
    /// (D-447). The uploader holds no
    /// logical-block state beyond the one window in flight.
    class Phase2WindowUploader {
    public:
        /// @brief Start the upload thread for one streamed device image.
        ///
        /// @details
        /// `device` must have begun its streamed upload; both references outlive
        /// the uploader, and the caller touches neither until `finish` returns.
        ///
        /// @param device Device image receiving the windows at their bases.
        /// @param sealing Retained host image receiving each window's sealing data.
        /// @return An idle uploader.
        Phase2WindowUploader(CudaPhase2ProjectionBuffer& device,
            Phase2ProjectionArena& sealing);

        /// @brief Destroy a finished uploader.
        ///
        /// @return Nothing.
        /// @invariant `finish` has joined the thread.
        ~Phase2WindowUploader();

        Phase2WindowUploader(const Phase2WindowUploader&) = delete;
        Phase2WindowUploader& operator=(const Phase2WindowUploader&) = delete;

        /// @brief Hand one complete window to the upload thread.
        ///
        /// @details
        /// Waits until the previous window is uploaded and retained, takes
        /// ownership of `window`, wakes the thread and releases the previous
        /// window on the calling thread.
        ///
        /// @param window Adopted window whose references already address their
        ///        complete-image offsets; non-null.
        /// @param bases Complete-image column positions of the window's first rows,
        ///        equal to the ends of every earlier window.
        /// @return Nothing.
        /// @invariant `finish` has not been called.
        void submit(std::unique_ptr<Phase2ProjectionArena> window,
            const Phase2ProjectionArena::ShardBases& bases);

        /// @brief Wait for the last window, release it and join the thread.
        ///
        /// @details
        /// After return the device image holds every submitted window and the
        /// sealing image their retained data; the caller may seal the upload.
        ///
        /// @return Nothing.
        /// @invariant Called once.
        void finish();

    private:
        /// @brief Thread body: upload and retain each submitted window in order.
        ///
        /// @details
        /// Waits for a pending window or the finish request; copies the window's
        /// used column ranges into the device image, appends its retained data to
        /// the sealing image, clears the pending flag and wakes the submitter.
        /// The finish request arrives only with no window pending.
        ///
        /// @return Nothing.
        void work();

        CudaPhase2ProjectionBuffer& device_;
        Phase2ProjectionArena& sealing_;
        std::mutex mutex_;
        std::condition_variable ready_;
        std::condition_variable uploaded_;
        std::unique_ptr<Phase2ProjectionArena> window_;
        Phase2ProjectionArena::ShardBases bases_{};
        bool pending_ = false;
        bool finished_ = false;
        std::thread thread_;
    };

    /// @brief Fixed-capacity CUDA ownership for reusable Phase 2 task inputs.
    ///
    /// @details
    /// Owns the four pointer-free task columns separately from the resident
    /// logical-block image because their lifetime is one executor sweep. All
    /// allocations occur in the constructor; upload copies bounded used prefixes,
    /// and checksum validates the complete device transfer without allocation.
    class CudaPhase2TaskBuffer {
    public:
        /// @brief Measure device and aligned arena bytes before creating a buffer.
        ///
        /// @details
        /// Validates the capacity shape and queries current CUDA library workspace
        /// without allocating device arrays. Includes every processing column and
        /// the 256-byte padding required by the common arena.
        ///
        /// @param capacity Proposed processing and input column capacities.
        /// @return Device payload bytes and exact aligned arena footprint.
        /// @invariant No buffer ownership or proof state is created or changed.
        static Phase2DeviceAllocationSize measureAllocation(Phase2TaskProjectionCapacity capacity);

        /// @brief Allocate every device task column once.
        ///
        /// @details
        /// Validates the CUDA device and positive capacities, then allocates fixed
        /// arrays for tasks, request batches, mandatory terms, and stumps plus one
        /// checksum word.
        ///
        /// @param fixedCapacity Immutable task-column element ceilings.
        /// @param allocation Optional shared arena; null owns standalone arrays.
        /// @return An empty device task buffer owning all allocations.
        /// @invariant No method changes an allocation address or capacity.
        explicit CudaPhase2TaskBuffer(
            Phase2TaskProjectionCapacity fixedCapacity, CudaPhase2Arena* allocation = nullptr);

        /// @brief Release every fixed device task allocation.
        ///
        /// @details
        /// Frees the checksum word and all four task arrays exactly once. Copy and
        /// move operations are disabled, so ownership cannot alias.
        ///
        /// @return Nothing.
        ~CudaPhase2TaskBuffer();

        CudaPhase2TaskBuffer(const CudaPhase2TaskBuffer&) = delete;
        CudaPhase2TaskBuffer& operator=(const CudaPhase2TaskBuffer&) = delete;
        CudaPhase2TaskBuffer(CudaPhase2TaskBuffer&&) = delete;
        CudaPhase2TaskBuffer& operator=(CudaPhase2TaskBuffer&&) = delete;

        /// @brief Upload one task arena's used prefixes without allocation.
        ///
        /// @details
        /// Asserts every used length against its startup ceiling and copies tasks,
        /// batches, terms, and stumps in canonical order. Empty prefixes issue no
        /// zero-byte CUDA operation.
        ///
        /// @param hostTasks Fixed-capacity host task image to copy.
        /// @return Nothing.
        /// @invariant Device used prefixes equal the host prefixes after return.
        void upload(const Phase2TaskProjectionArena& hostTasks);

        /// @brief Hash all uploaded task bytes on the CUDA device.
        ///
        /// @details
        /// Uses one deterministic device thread and the same FNV-1a transfer twin
        /// as the resident projection, restricted to the four task columns.
        ///
        /// @return Device-computed checksum of every uploaded used prefix.
        /// @invariant At least one task descriptor has been uploaded.
        uint64_t launchChecksum() const;

        /// @brief Report every fixed task-column allocation and checksum word.
        ///
        /// @details
        /// Returns the exact sum of the four immutable task-column allocation
        /// requests and the reusable checksum-word allocation captured during
        /// construction. Per-pass uploads cannot change the result.
        ///
        /// @return Exact startup bytes owned by this task buffer.
        /// @invariant The value is constant for the object's lifetime.
        uint64_t fixedAllocationBytes() const;

    private:
        bool ownsDeviceAllocations_{ true };
        friend class CudaPhase2GrowthBuffer;
        friend class CudaPhase2OrderingBuffer;
        friend class CudaPhase2EvaluationBuffer;
        static constexpr uint32_t kTaskColumnCount = 4;
        Phase2TaskProjectionCapacity capacity_{};
        void* deviceColumns_[kTaskColumnCount]{};
        void* deviceChecksum_{ nullptr };
        uint32_t usedCounts_[kTaskColumnCount]{};
        uint64_t fixedAllocationBytes_{ 0 };
    };

    /// @brief Fixed-capacity CUDA ownership for global statement filter/sort.
    ///
    /// @details
    /// Owns original and exact-class schedules, original-to-class indices,
    /// per-class and per-call counts and offsets, two compact retained-key arrays,
    /// and one reusable CUB allocation. Count and emit launches process only exact
    /// classes; original calls map to shared compact spans before growth. A segmented
    /// radix sort orders each class span by full-width name rank and statement.
    class CudaPhase2FilterSortBuffer {
    public:
        /// @brief Measure device and aligned arena bytes before creating a buffer.
        ///
        /// @details
        /// Validates the capacity shape and queries current CUDA library workspace
        /// without allocating device arrays. Includes every processing column and
        /// the 256-byte padding required by the common arena.
        ///
        /// @param capacity Proposed processing and input column capacities.
        /// @return Device payload bytes and exact aligned arena footprint.
        /// @invariant No buffer ownership or proof state is created or changed.
        static Phase2DeviceAllocationSize measureAllocation(Phase2FilterScheduleCapacity capacity);

        /// @brief Allocate all global filter/sort device storage once.
        ///
        /// @details
        /// Validates the CUDA device and measured ceilings, queries scan and radix
        /// temporary sizes, then allocates every original-call, class, mapping,
        /// count, offset, key, and scratch array. Later sweeps overwrite only
        /// bounded used prefixes.
        ///
        /// @param fixedCapacity Immutable call, examined-row, retained-row, and
        ///                      per-call ceilings.
        /// @param allocation Optional shared arena; null owns standalone arrays.
        /// @return An empty device filter/sort owner.
        /// @invariant No method changes an allocation address or capacity.
        explicit CudaPhase2FilterSortBuffer(
            Phase2FilterScheduleCapacity fixedCapacity, CudaPhase2Arena* allocation = nullptr);

        /// @brief Release every fixed filter/sort device allocation.
        ///
        /// @details
        /// Frees the shared CUB scratch, both retained-key arrays, original and
        /// class count/offset columns, mapping, and both schedules exactly once.
        ///
        /// @return Nothing.
        ~CudaPhase2FilterSortBuffer();

        CudaPhase2FilterSortBuffer(
            const CudaPhase2FilterSortBuffer&) = delete;
        CudaPhase2FilterSortBuffer& operator=(
            const CudaPhase2FilterSortBuffer&) = delete;
        CudaPhase2FilterSortBuffer(CudaPhase2FilterSortBuffer&&) = delete;
        CudaPhase2FilterSortBuffer& operator=(
            CudaPhase2FilterSortBuffer&&) = delete;

        /// @brief Filter and sort every scheduled call on the CUDA device.
        ///
        /// @details
        /// Uploads original calls, exact classes, and the original-to-class map;
        /// counts each class's processor-accepted first-8,192 rows; scans and emits
        /// one stable compact class span; radix-sorts each class span; and maps
        /// original calls to their class count and offset. No processor semantic
        /// gate is executed on the host.
        ///
        /// @param projection Uploaded resident semantic projection.
        /// @param schedule Host schedule in processor call order.
        /// @return Total retained statement rows across all calls.
        /// @invariant Sorted keys encode full-width decoded-name rank and ascending
        ///            original statement index; every original call names its
        ///            exact class span.
        uint32_t filterAndSort(
            const CudaPhase2ProjectionBuffer& projection,
            const Phase2FilterScheduleArena& schedule);

        /// @brief Download the most recent per-call retained counts.
        ///
        /// @details
        /// Copies exactly the last uploaded call prefix into caller-owned storage.
        /// It is a transfer twin and later integration diagnostic, not a semantic
        /// processor replay.
        ///
        /// @param destination Caller-owned output array.
        /// @param capacity Destination element capacity.
        /// @return Number of counts copied.
        /// @invariant `capacity` covers every used call.
        uint32_t downloadCallCounts(
            uint32_t* destination, uint32_t capacity) const;

        /// @brief Download the most recent class-segmented composite keys.
        ///
        /// @details
        /// Copies the compact retained prefix. Each key is ordered and encoded as
        /// `(decoded-name rank, original statement index)` within each class, using the public
        /// bit constants in `phase2_projection.hpp`.
        ///
        /// @param destination Caller-owned output array.
        /// @param capacity Destination element capacity.
        /// @return Number of keys copied.
        /// @invariant `capacity` covers every retained key.
        uint32_t downloadSortedKeys(
            uint64_t* destination, uint32_t capacity) const;

        /// @brief Report every fixed filter allocation including CUB scratch.
        ///
        /// @details
        /// Returns the exact original-call, class, mapping, count, offset,
        /// double-key, and maximum reusable scan-or-sort scratch allocation sum
        /// captured during construction. Filtering cannot change the total.
        ///
        /// @return Exact startup bytes owned by this filter buffer.
        /// @invariant The value is constant for the object's lifetime.
        uint64_t fixedAllocationBytes() const;

    private:
        bool ownsDeviceAllocations_{ true };
        friend class CudaPhase2GrowthBuffer;
        Phase2FilterScheduleCapacity capacity_{};
        void* deviceCalls_{ nullptr };
        void* deviceClasses_{ nullptr };
        void* deviceCallClassIndices_{ nullptr };
        void* deviceCounts_{ nullptr };
        void* deviceOffsets_{ nullptr };
        void* deviceClassCounts_{ nullptr };
        void* deviceClassOffsets_{ nullptr };
        void* deviceKeysInput_{ nullptr };
        void* deviceKeysOutput_{ nullptr };
        void* deviceTemporary_{ nullptr };
        std::size_t scanTemporaryBytes_{ 0 };
        std::size_t sortTemporaryBytes_{ 0 };
        uint32_t usedCalls_{ 0 };
        uint32_t usedClasses_{ 0 };
        uint32_t usedRetainedRows_{ 0 };
        uint64_t fixedAllocationBytes_{ 0 };
    };

    /// @brief Fixed-capacity CUDA ownership for normalized-key frontier growth.
    ///
    /// @details
    /// Owns the request-call schedule, mandatory direct/suffix masks, two
    /// ping-pong node frontiers, one immutable prefix header per live node, three
    /// measured prefix-value pools, bounded short/long and canonical node index
    /// lists, deterministic candidate-window records and compact survivor
    /// ordinals, persistent accepted-event and raw-request arrays, and shared
    /// semantic/observation counters. Frontier nodes store compact filtered-
    /// position paths; persistent events store statement-index paths. All
    /// allocation occurs in the constructor.
    class CudaPhase2GrowthBuffer {
    public:
        /// @brief Measure device and aligned arena bytes before creating a buffer.
        ///
        /// @details
        /// Validates the capacity shape and queries current CUDA library workspace
        /// without allocating device arrays. Includes every processing column and
        /// the 256-byte padding required by the common arena.
        ///
        /// @param capacity Proposed processing and input column capacities.
        /// @return Device payload bytes and exact aligned arena footprint.
        /// @invariant No buffer ownership or proof state is created or changed.
        static Phase2DeviceAllocationSize measureAllocation(Phase2GrowthCapacity capacity);

        /// @brief Allocate every measured request-growth array once.
        ///
        /// @details
        /// Validates the CUDA contract and positive retained FTA ceilings, then
        /// allocates the call schedule, two retained-row masks, two equal node
        /// frontiers, one frontier-sized prefix-header array, three measured
        /// prefix-value pools, short/long and canonical node index lists, semantic
        /// sort/scan columns, one bounded candidate-attempt window with flags and
        /// survivor ordinals, one accepted-event ledger, one raw-request array,
        /// and one counter record, including the exact remaining-span census.
        /// Later growth passes reset used counters but never change these addresses
        /// or capacities.
        ///
        /// @param fixedCapacity Immutable frontier, event, and request ceilings.
        /// @param allocation Optional shared arena; null owns standalone arrays.
        /// @return An empty device growth owner.
        /// @invariant No method changes an allocation address or capacity.
        explicit CudaPhase2GrowthBuffer(Phase2GrowthCapacity fixedCapacity, CudaPhase2Arena* allocation = nullptr);

        /// @brief Release every fixed request-growth device allocation.
        ///
        /// @details
        /// Frees counters, raw requests, accepted events, candidate windows,
        /// semantic sort/scan columns, index lists, prefix pools and headers, both
        /// frontiers, masks, and the call schedule in reverse ownership order.
        /// Copy and move operations are disabled.
        ///
        /// @return Nothing.
        ~CudaPhase2GrowthBuffer();

        CudaPhase2GrowthBuffer(const CudaPhase2GrowthBuffer&) = delete;
        CudaPhase2GrowthBuffer& operator=(
            const CudaPhase2GrowthBuffer&) = delete;
        CudaPhase2GrowthBuffer(CudaPhase2GrowthBuffer&&) = delete;
        CudaPhase2GrowthBuffer& operator=(CudaPhase2GrowthBuffer&&) = delete;

        /// @brief Run bulk normalized-key request frontier growth.
        ///
        /// @details
        /// Uploads the pointer-free growth schedule, builds mandatory-view and
        /// suffix masks over the already filtered spans, seeds unsplit roots or
        /// exact stump nodes, and expands all live nodes through two global
        /// ping-pong frontiers. Short spans stay one-thread-per-node; long spans
        /// are compacted into the bounded index list and distribute candidates
        /// across one cooperative block. Kernels append only capacity-bounded
        /// semantic events and pre-dedup whole-key requests; processor ordering
        /// and evaluation are separate later stages.
        ///
        /// @param projection Uploaded resident logical-block image.
        /// @param tasks Uploaded executor task, batch, term, and stump image.
        /// @param filter Completed global filter/sort result for these calls.
        /// @param schedule Host request-growth schedule in processor batch order.
        /// @param parameters Immutable analyzer request-shape limits.
        /// @return Used event/request counts, maximum live frontier/cooperative
        ///         sizes, and the observation-only remaining-candidate census.
        /// @invariant Every schedule link names compatible uploaded task, batch,
        ///            filter, and logical-block records.
        Phase2GrowthResult runRequestGrowth(
            const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks,
            const CudaPhase2FilterSortBuffer& filter,
            const Phase2GrowthScheduleArena& schedule,
            DevicePhase2GrowthParameters parameters);

        /// @brief Generate complete requests through bounded GPU work windows.
        ///
        /// @details
        /// A pass-local device frontier store holds the logical frontier: one
        /// buffer the level being grown, the other its children. Pages of a level
        /// are read where they lie; only rows of a level that outgrows a buffer
        /// spill to host static-pool pages, which later upload through two
        /// alternating registered staging blocks into the consumed buffer. CUDA
        /// performs seed, prefix, candidate and exact-probe work; the host keeps
        /// upper bounds of the append counters (one node, event and request per
        /// seed pair or survivor) and reads the exact counts only when a bound
        /// could overflow, at prefix windows and at level ends. Full request
        /// storage calls the synchronous drain before any event slot is reused;
        /// the drain keeps pending store rows. The level-end cutoff prunes host
        /// pages and device rows alike. Nothing in the store outlives the call.
        ///
        /// @param projection Frozen device logical-block image.
        /// @param tasks Uploaded task image.
        /// @param hostTasks Host twin, read only for scheduling sizes.
        /// @param filter Completed filter spans for all scheduled calls.
        /// @param schedule Original generator calls in this physical task image.
        /// @param parameters Semantic request limits; census mode must be zero.
        /// @param drain Synchronous consumer; true commits a completed logical level, even when empty.
        /// @param context Opaque caller-owned drain context.
        /// @param taskSubkeyCounts Output cumulative accepted-subkey totals.
        /// @param taskCapacity Number of caller-owned task counter slots.
        /// @param retainedCalls Optional cutoff query run after each completed
        ///        level's drain: fills one keep flag per scheduled call (0 stops
        ///        the call's descendants) and returns whether any flag is 0.
        /// @return Logical totals and frontier transfer/backing measurements.
        /// @invariant One prefix fits minimum execution scratch; request reuse
        ///            occurs only after its synchronous drain returns.
        Phase2GrowthStreamResult runRequestGrowthWindows(
            const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks,
            const Phase2TaskProjectionArena& hostTasks,
            const CudaPhase2FilterSortBuffer& filter,
            const Phase2GrowthScheduleArena& schedule,
            DevicePhase2GrowthParameters parameters,
            void (*drain)(void*, const CudaPhase2GrowthBuffer&, bool), void* context,
            uint64_t* taskSubkeyCounts, uint32_t taskCapacity,
            bool (*retainedCalls)(void*, uint8_t*, uint32_t) = nullptr);

        /// @brief Download the most recent accepted growth-event ledger.
        ///
        /// @details
        /// Copies the device append prefix in arbitrary execution order. Each
        /// record is self-identifying by call, run, count, and statement path, so
        /// tests and the later ordering pass do not depend on append order.
        ///
        /// @param destination Caller-owned event output array.
        /// @param capacity Destination element capacity.
        /// @return Number of event records copied.
        /// @invariant `capacity` covers the last result's accepted-event count.
        uint32_t downloadAcceptedEvents(
            DeviceAcceptedGrowthEvent* destination,
            uint32_t capacity) const;

        /// @brief Download a bounded consecutive metadata window.
        ///
        /// @details Copies retained GPU decisions without changing ownership
        /// or imposing a whole-buffer contiguous host allocation.
        ///
        /// @param begin First retained row, including the end for an empty range.
        /// @param count Number of consecutive rows to copy.
        /// @param destination Caller-owned space for count rows; null only if empty.
        /// @return Nothing.
        /// @invariant The requested range lies in the last completed device image.
        void downloadAcceptedEventWindow(uint32_t begin, uint32_t count, DeviceAcceptedGrowthEvent* destination) const;

        /// @brief Download the most recent raw whole-key request ledger.
        ///
        /// @details
        /// Copies every pre-dedup request hit. `growthPosition` remains zero until
        /// exact processor event order is reconstructed in the next port step.
        ///
        /// @param destination Caller-owned raw-request output array.
        /// @param capacity Destination element capacity.
        /// @return Number of request records copied.
        /// @invariant `capacity` covers the last result's raw-request count.
        uint32_t downloadRawRequests(
            DeviceRawGrowthRequest* destination,
            uint32_t capacity) const;

        /// @brief Download exact processor split-work tallies by task.
        ///
        /// @details
        /// Copies one count per uploaded executor task. A count is incremented
        /// only for a growth node whose normalized key passed the subkey owner
        /// probe, exactly matching `ExpressionAnalyzer::g_growthMatchCount`.
        /// Whole-key-only events retained for request evaluation do not enter
        /// this statistic.
        ///
        /// @param destination Caller-owned task-count output array.
        /// @param capacity Destination element capacity.
        /// @return Number of task counts copied.
        /// @invariant `runRequestGrowth` completed and `capacity` covers the
        ///            uploaded task count.
        uint32_t downloadTaskSubkeyCounts(
            uint32_t* destination,
            uint32_t capacity) const;

        /// @brief Report the immutable bytes owned by all growth arrays.
        ///
        /// @details
        /// Computes the schedule, two mandatory-mask, two frontier, prefix-header,
        /// three prefix-value, all node sort/scan and candidate-window columns,
        /// accepted-event, raw-request, per-task subkey-count, and shared
        /// semantic/census counter allocations from constructor capacities and
        /// public record sizes. The value excludes resident projection, task,
        /// filter, and later ordering scratch.
        ///
        /// @return Exact startup allocation bytes owned by this growth buffer.
        /// @invariant The result is constant for the object's lifetime.
        uint64_t fixedAllocationBytes() const;

    private:
        bool ownsDeviceAllocations_{ true };
        friend class CudaPhase2OrderingBuffer;
        friend class CudaPhase2EvaluationBuffer;
        Phase2GrowthCapacity capacity_{};
        void* deviceCalls_{ nullptr };
        void* deviceViewMasks_{ nullptr };
        void* deviceSuffixMasks_{ nullptr };
        void* deviceFrontiers_[2]{};
        void* deviceKeepByCall_{ nullptr };
        void* devicePrefixes_{ nullptr };
        void* devicePrefixPayload_{ nullptr };
        void* devicePrefixVariables_{ nullptr };
        void* devicePrefixSecondary_{ nullptr };
        void* deviceShortNodeFlags_{ nullptr };
        void* deviceCooperativeNodeFlags_{ nullptr };
        void* deviceShortNodeIndices_{ nullptr };
        void* deviceCooperativeNodeIndices_{ nullptr };
        void* deviceCanonicalNodeIndices_[2]{};
        void* deviceNodeSortKeys_[2]{};
        void* deviceNodeAttemptStarts_{ nullptr };
        void* deviceNodeAttemptEnds_{ nullptr };
        void* deviceCandidateAttempts_{ nullptr };
        void* deviceCandidateFlags_{ nullptr };
        void* deviceCandidateSurvivorIndices_{ nullptr };
        void* deviceSelectionTemporary_{ nullptr };
        void* deviceAcceptedEvents_{ nullptr };
        void* deviceRawRequests_{ nullptr };
        void* deviceTaskSubkeyCounts_{ nullptr };
        void* deviceCounters_{ nullptr };
        uint32_t usedAcceptedEvents_{ 0 };
        uint32_t usedRawRequests_{ 0 };
        uint32_t usedTaskCount_{ 0 };
        std::size_t selectionTemporaryBytes_{ 0 };
    };

    /// @brief Fixed-capacity CUDA ownership for exact event and request order.
    ///
    /// @details
    /// Owns reusable radix-sort keys/indices, the preserved ordered event index,
    /// segmented growth-position scan arrays, two request-token arrays,
    /// a count word, and one shared CUB scratch
    /// allocation. No ordering call allocates or falls back to the processor.
    class CudaPhase2OrderingBuffer {
    public:
        /// @brief Measure device and aligned arena bytes before creating a buffer.
        ///
        /// @details
        /// Validates the capacity shape and queries current CUDA library workspace
        /// without allocating device arrays. Includes every processing column and
        /// the 256-byte padding required by the common arena.
        ///
        /// @param capacity Proposed processing and input column capacities.
        /// @return Device payload bytes and exact aligned arena footprint.
        /// @invariant No buffer ownership or proof state is created or changed.
        static Phase2DeviceAllocationSize measureAllocation(Phase2OrderingCapacity capacity);

        /// @brief Allocate all exact-order scratch once.
        ///
        /// @details
        /// Validates positive ceilings,
        /// queries the maximum radix/scan temporary requirement, then allocates
        /// every fixed array and the shared CUB scratch region.
        ///
        /// @param fixedCapacity Immutable event and request ceilings.
        /// @param allocation Optional shared arena; null owns standalone arrays.
        /// @return An empty reusable ordering owner.
        /// @invariant No later method changes an allocation address or capacity.
        explicit CudaPhase2OrderingBuffer(
            Phase2OrderingCapacity fixedCapacity, CudaPhase2Arena* allocation = nullptr);

        /// @brief Release every exact-order device allocation.
        ///
        /// @details
        /// Frees shared scratch, counters, request tokens,
        /// scan arrays, the preserved order index, and radix arrays exactly once.
        ///
        /// @return Nothing.
        ~CudaPhase2OrderingBuffer();

        CudaPhase2OrderingBuffer(
            const CudaPhase2OrderingBuffer&) = delete;
        CudaPhase2OrderingBuffer& operator=(
            const CudaPhase2OrderingBuffer&) = delete;
        CudaPhase2OrderingBuffer(CudaPhase2OrderingBuffer&&) = delete;
        CudaPhase2OrderingBuffer& operator=(
            CudaPhase2OrderingBuffer&&) = delete;

        /// @brief Reconstruct event order and retain every recordable request.
        ///
        /// @details
        /// Stable radix passes order task, call, stump and statement paths. A
        /// segmented scan assigns task-local growth positions. Every recordable
        /// event emits its own token, including repeated semantic request keys;
        /// tokens are sorted by event order before evaluation.
        ///
        /// @param projection Uploaded resident logical-block image.
        /// @param tasks Uploaded task and batch image used by the growth calls.
        /// @param growth Completed unordered growth content.
        /// @return Ordered event and request counts.
        /// @invariant Growth content and its uploaded call schedule are unchanged.
        Phase2OrderingResult orderRequests(
            const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks,
            const CudaPhase2GrowthBuffer& growth);

        /// @brief Download the exact ordered event-index permutation.
        ///
        /// @param destination Caller-owned event-index output.
        /// @param capacity Destination element capacity.
        /// @return Number of ordered indices copied.
        /// @invariant `capacity` covers the last ordered event count.
        uint32_t downloadOrderedEventIndices(
            uint32_t* destination,
            uint32_t capacity) const;

        /// @brief Download request tokens in exact processor stream order.
        ///
        /// @param destination Caller-owned ordered-token output.
        /// @param capacity Destination element capacity.
        /// @return Number of request tokens copied.
        /// @invariant `capacity` covers the last request count.
        uint32_t downloadOrderedRequests(
            DeviceOrderedRequestToken* destination,
            uint32_t capacity) const;

        /// @brief Download a bounded consecutive metadata window.
        ///
        /// @details Copies retained GPU decisions without changing ownership
        /// or imposing a whole-buffer contiguous host allocation.
        ///
        /// @param begin First retained row, including the end for an empty range.
        /// @param count Number of consecutive rows to copy.
        /// @param destination Caller-owned space for count rows; null only if empty.
        /// @return Nothing.
        /// @invariant The requested range lies in the last completed device image.
        void downloadOrderedRequestWindow(uint32_t begin, uint32_t count, DeviceOrderedRequestToken* destination) const;

        /// @brief Report every fixed ordering allocation including CUB scratch.
        ///
        /// @return Exact startup bytes owned by this ordering buffer.
        /// @invariant The value is constant for the object's lifetime.
        uint64_t fixedAllocationBytes() const;

    private:
        bool ownsDeviceAllocations_{ true };
        friend class CudaPhase2EvaluationBuffer;
        Phase2OrderingCapacity capacity_{};
        void* deviceKeys_[2]{};
        void* deviceIndices_[2]{};
        void* deviceOrderedIndices_{ nullptr };
        void* deviceScanValues_[2]{};
        void* deviceRequestTokens_[2]{};
        void* deviceRequestCount_{ nullptr };
        void* deviceTemporary_{ nullptr };
        std::size_t temporaryBytes_{ 0 };
        uint32_t usedEvents_{ 0 };
        uint32_t usedRequests_{ 0 };
    };


    /// @brief Fixed-capacity CUDA ownership for request evaluation and output.
    ///
    /// @details
    /// Owns request states and scan columns, reverse-owner work, compact subset
    /// candidates, encoded hits, LocalMemoryValue work, complete firing headers,
    /// every variable provenance/output arena, canonical index arrays, per-block
    /// doom lines and trigger indices, selection counters, semantic counters, and
    /// one maximum-sized reusable CUB scratch allocation. The first
    /// live stage expands exact ordered requests through LocalMemoryValue work;
    /// later stages fill the already-owned firing output columns.
    class CudaPhase2EvaluationBuffer {
    public:
        /// @brief Measure device and aligned arena bytes before creating a buffer.
        ///
        /// @details
        /// Validates the capacity shape and queries current CUDA library workspace
        /// without allocating device arrays. Includes every processing column and
        /// the 256-byte padding required by the common arena.
        ///
        /// @param capacity Proposed processing and input column capacities.
        /// @return Device payload bytes and exact aligned arena footprint.
        /// @invariant No buffer ownership or proof state is created or changed.
        static Phase2DeviceAllocationSize measureAllocation(Phase2EvaluationCapacity capacity);

        /// @brief Allocate every retained evaluator work and output array once.
        ///
        /// @details
        /// Validates all next-power-of-two ceilings, queries scan and selection
        /// temporary requirements, then allocates every work, firing, provenance,
        /// byte, ordering, doom, counter, and shared-scratch column. No evaluation
        /// call allocates.
        ///
        /// @param fixedCapacity Immutable evaluator work and output ceilings.
        /// @param allocation Optional shared arena; null owns standalone arrays.
        /// @return An empty reusable evaluator owner.
        /// @invariant No later method changes an allocation address or capacity.
        explicit CudaPhase2EvaluationBuffer(
            Phase2EvaluationCapacity fixedCapacity, CudaPhase2Arena* allocation = nullptr);

        /// @brief Release every fixed evaluator allocation exactly once.
        ///
        /// @details
        /// Frees shared scratch, counters, doom arrays, ordering/output arenas,
        /// firing records, work arrays, flags, offsets, counts, and request states
        /// in reverse ownership order. Copy and move operations are disabled.
        ///
        /// @return Nothing.
        ~CudaPhase2EvaluationBuffer();

        CudaPhase2EvaluationBuffer(
            const CudaPhase2EvaluationBuffer&) = delete;
        CudaPhase2EvaluationBuffer& operator=(
            const CudaPhase2EvaluationBuffer&) = delete;
        CudaPhase2EvaluationBuffer(
            CudaPhase2EvaluationBuffer&&) = delete;
        CudaPhase2EvaluationBuffer& operator=(
            CudaPhase2EvaluationBuffer&&) = delete;

        /// @brief Expand exact ordered requests through projected LMV work.
        ///
        /// @details
        /// Applies known-statement dependency checks, premise validity folding,
        /// hypothesis and closed-scope gates, reverse-index expansion, remaining-
        /// argument subset compaction, ignore-u mapped normalized-key lookup, and
        /// LocalMemoryValue-run expansion entirely on the device. The resulting
        /// work prefix is the direct input to firing-content evaluation.
        ///
        /// @param projection Uploaded resident logical-block image.
        /// @param tasks Uploaded executor task image.
        /// @param growth Completed request-growth content and call schedule.
        /// @param ordering Exact ordered request tokens.
        /// @return Exact used counts at every evaluation expansion boundary.
        /// @invariant Every input belongs to the same uploaded Phase 2 sweep and
        ///            no processor semantic gate is replayed.
        Phase2EvaluationResult expandEvaluationWork(
            const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks,
            const CudaPhase2GrowthBuffer& growth,
            const CudaPhase2OrderingBuffer& ordering);

        /// @brief Materialize exact substituted firing expressions on the GPU.
        ///
        /// @details
        /// Reconstructs each selected LMV's reverse substitution from its ordered
        /// request and remaining-argument owner, applies the processor's greedy-
        /// longest token replacement and repeated token-leading `u_` removal,
        /// folds LMV validity into the request consensus, repeats the closed-scope
        /// filter for a deeper rule scope, and applies marker/demand purity plus
        /// ordis2 iteration-depth gates. Surviving rows receive a firing header and
        /// byte-exact expression slice in fixed device arenas. This stage records
        /// source/type fields, sorted-unique levels, and complete head provenance;
        /// final head verdicts, marker admission parameters, marker keys,
        /// remaining arguments, and marker-head argument slices are all filled
        /// in the same fixed output owner.
        ///
        /// @param projection Uploaded resident logical-block image.
        /// @param tasks Uploaded executor task image.
        /// @param growth Completed request-growth content and call schedule.
        /// @param ordering Exact ordered request tokens.
        /// @param valueBegin First prepared value to materialize.
        /// @param valueCount Maximum prepared values to materialize.
        /// @param markerKeyBegin First marker key included in each value.
        /// @param markerKeyCount Maximum marker keys included in each value.
        /// @return Exact firing-header and every variable-output used count.
        /// @invariant A value window was prepared for these same inputs; measured
        ///            bounds include scratch consumed before late rejection gates.
        Phase2FiringExpressionResult materializeFiringExpressions(
            const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks,
            const CudaPhase2GrowthBuffer& growth,
            const CudaPhase2OrderingBuffer& ordering, uint32_t valueBegin = 0,
            uint32_t valueCount = ~uint32_t{0}, uint32_t markerKeyBegin = 0,
            uint32_t markerKeyCount = ~uint32_t{0});

        /// @brief Measure a value window without allocating its output payloads.
        ///
        /// @details
        /// GPU-side measurement bounds every output column. Marker-key slicing
        /// permits one marker record to be emitted across several downloads.
        ///
        /// @param projection Frozen uploaded LB input.
        /// @param tasks Live task descriptors.
        /// @param growth Live request events and call descriptors.
        /// @param ordering Live request tokens.
        /// @param valueBegin First prepared value to measure.
        /// @param valueCount Number of values, at most outputPlanningRows.
        /// @param destination Caller-owned output bounds.
        /// @param markerKeyBegin First marker key, zero for complete records.
        /// @param markerKeyCount Maximum marker keys per record.
        /// @return Nothing.
        /// @invariant Measurement does not consume work or change proof state.
        void measureFiringOutputBounds(const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks, const CudaPhase2GrowthBuffer& growth,
            const CudaPhase2OrderingBuffer& ordering, uint32_t valueBegin,
            uint32_t valueCount, Phase2OutputUsage* destination,
            uint32_t markerKeyBegin = 0, uint32_t markerKeyCount = ~uint32_t{0});

        /// @brief Maximum metadata rows in one bounded output-planning transfer.
        static constexpr uint32_t outputPlanningRows = 1024;

        /// @brief Materialize the next capacity-reserved firing window.
        ///
        /// @details
        /// Packs consecutive complete values across bounded measurement pages
        /// until an output column fills; metadata pages do not force downloads.
        /// An oversized marker is sliced by keys. When even the next fragment
        /// needs more room, reports its requirements without consuming work;
        /// the allocation controller must lend capacity or segment that payload.
        ///
        /// @param projection Frozen uploaded LB input.
        /// @param tasks Live task descriptors.
        /// @param growth Live request events and call identities.
        /// @param ordering Live request tokens.
        /// @param cursor In/out next value/key position in this prepared window.
        /// @return Completed-window, complete-stream or capacity-request state.
        /// @invariant The caller retains ready output before the next call, and
        ///            no capacity request changes the continuation position.
        Phase2FiringWindow nextFiringWindow(const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks, const CudaPhase2GrowthBuffer& growth,
            const CudaPhase2OrderingBuffer& ordering, Phase2FiringCursor& cursor);

        /// @brief Begin evaluation of one frozen request-buffer image.
        ///
        /// @details
        /// Performs request gates once and records 64-bit owner-run offsets.
        /// The request image remains live until all value windows are consumed.
        ///
        /// @param projection Immutable uploaded LB input.
        /// @param tasks Immutable task descriptors.
        /// @param growth Live request events and call identities.
        /// @param ordering Unique request tokens retained in VRAM.
        /// @param metrics Optional observation-only expansion totals and occupancy.
        /// @return Nothing.
        /// @invariant The previous streaming evaluation is exhausted.
        void beginEvaluationWindows(const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks, const CudaPhase2GrowthBuffer& growth,
            const CudaPhase2OrderingBuffer& ordering,
            Phase2EvaluationStreamMetrics* metrics = nullptr);

        /// @brief Prepare the next capacity-bounded rule-value window.
        ///
        /// @details
        /// Owner windows fit candidate and encoded-hit storage in the worst
        /// case. An encoded hit's values may span any number of windows. True
        /// exposes values for materialization; false means the complete input
        /// is exhausted, including the defined no-hit case. Previously exposed
        /// output must be retained before this method reuses working storage.
        ///
        /// @param projection Same immutable LB input used at begin.
        /// @param tasks Same immutable task descriptors used at begin.
        /// @param growth Same live request events used at begin.
        /// @param ordering Same request tokens used at begin.
        /// @return Whether a nonempty value window is ready for consumption.
        /// @invariant Each owner and each selected value is visited exactly once.
        bool nextEvaluationValues(const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks, const CudaPhase2GrowthBuffer& growth,
            const CudaPhase2OrderingBuffer& ordering);

        /// @brief Canonically order complete device firing records by content.
        ///
        /// @details
        /// Builds an index permutation over the last materialized firing prefix.
        /// Logical-block index groups independent identifier namespaces; within
        /// each block the comparator is the exact `applyFiringRecords` content
        /// order for heads, markers, and ordis2 demands. Iterative parallel merge
        /// passes use two fixed index arrays and never move variable payloads.
        ///
        /// @param projection Uploaded resident image that decodes every local
        ///                   name and rule-interner identifier in the records.
        /// @return Number of canonically ordered firing indices.
        /// @invariant `materializeFiringExpressions` completed against the same
        ///            projection and every record belongs to its stated block.
        uint32_t orderFiringRecords(
            const CudaPhase2ProjectionBuffer& projection);

        /// @brief Attach original GPU early-exit verdicts to the current output.
        ///
        /// @details The deactivation predicate is unchanged. Selection is deferred
        /// until original request positions are known across all capacity drains.
        ///
        /// @param projection Frozen logical-block image used for materialization.
        /// @return Nothing.
        /// @invariant Every retained header refers to the current generated bytes.
        void markDoomTriggers(const CudaPhase2ProjectionBuffer& projection);

        /// @brief Select exact doom winners and compact their retained prefixes.
        ///
        /// @details
        /// Mirrors `burstDeactivates` for every materialized head, reduces the
        /// lexicographically earliest `(growth position, part ordinal)` trigger
        /// per logical block, and stably compacts the canonical firing-order
        /// permutation. A doomed block keeps only its winning part through the
        /// first triggering ordered request at that growth position; an undoomed
        /// block keeps every record.
        ///
        /// @param projection Uploaded resident image containing roles, goals,
        ///                   known statements, names, and compressor mode.
        /// @return Number of canonical firing indices retained for sealing.
        /// @invariant `orderFiringRecords` completed against this projection and
        ///            every retained prefix is timing-independent.
        uint32_t selectDoomPrefixes(
            const CudaPhase2ProjectionBuffer& projection);

        /// @brief Download compact subset-surviving evaluation candidates.
        ///
        /// @param destination Caller-owned candidate output.
        /// @param capacity Destination element capacity.
        /// @return Number of compact candidates copied.
        /// @invariant `capacity` covers the last candidate-owner count.
        uint32_t downloadCandidates(
            DeviceEvaluationCandidate* destination,
            uint32_t capacity) const;

        /// @brief Download compact encoded-hit candidates.
        ///
        /// @param destination Caller-owned encoded-hit output.
        /// @param capacity Destination element capacity.
        /// @return Number of encoded-hit records copied.
        /// @invariant `capacity` covers the last encoded-hit count.
        uint32_t downloadEncodedHits(
            DeviceEvaluationCandidate* destination,
            uint32_t capacity) const;

        /// @brief Download LocalMemoryValue work selected by the last expansion.
        ///
        /// @param destination Caller-owned value-work output.
        /// @param capacity Destination element capacity.
        /// @return Number of LocalMemoryValue references copied.
        /// @invariant `capacity` covers the last local-value count.
        uint32_t downloadValueWork(
            DeviceEvaluationValueWork* destination,
            uint32_t capacity) const;

        /// @brief Download firing headers from the last expression materialization.
        ///
        /// @param destination Caller-owned firing-header output.
        /// @param capacity Destination element capacity.
        /// @return Number of firing headers copied.
        /// @invariant `capacity` covers the last firing-record count.
        uint32_t downloadFiringRecords(
            DevicePhase2FiringRecord* destination,
            uint32_t capacity) const;

        /// @brief Download generated bytes from the last expression materialization.
        ///
        /// @param destination Caller-owned byte output.
        /// @param capacity Destination byte capacity.
        /// @return Number of generated bytes copied.
        /// @invariant `capacity` covers the last generated-byte count.
        uint32_t downloadGeneratedBytes(
            char* destination,
            uint32_t capacity) const;

        /// @brief Download sorted-unique level values from the last materialization.
        ///
        /// @param destination Caller-owned integer output.
        /// @param capacity Destination element capacity.
        /// @return Number of level values copied.
        /// @invariant `capacity` covers the last level-value count.
        uint32_t downloadLevelValues(
            int32_t* destination,
            uint32_t capacity) const;

        /// @brief Download head provenance dependencies from the last materialization.
        ///
        /// @param destination Caller-owned dependency output.
        /// @param capacity Destination element capacity.
        /// @return Number of dependencies copied.
        /// @invariant `capacity` covers the last origin-dependency count.
        uint32_t downloadOriginDependencies(
            DeviceEvaluationDependency* destination,
            uint32_t capacity) const;

        /// @brief Download transformed marker key slices in LMV order.
        ///
        /// @param destination Caller-owned byte-slice output.
        /// @param capacity Destination element capacity.
        /// @return Number of marker key slices copied.
        /// @invariant `capacity` covers the last marker-key count and every slice
        ///            addresses the last generated-byte prefix.
        uint32_t downloadMarkerKeys(
            DeviceEvaluationByteSlice* destination,
            uint32_t capacity) const;

        /// @brief Download sorted-unique marker remaining-argument identifiers.
        ///
        /// @param destination Caller-owned rule-interner identifier output.
        /// @param capacity Destination element capacity.
        /// @return Number of remaining-argument identifiers copied.
        /// @invariant `capacity` covers the last marker-remaining count.
        uint32_t downloadMarkerRemainingArgs(
            int32_t* destination,
            uint32_t capacity) const;

        /// @brief Download sorted-unique non-marker head-argument slices.
        ///
        /// @param destination Caller-owned byte-slice output.
        /// @param capacity Destination element capacity.
        /// @return Number of marker argument slices copied.
        /// @invariant `capacity` covers the last marker-argument count and every
        ///            slice addresses the last generated-byte prefix.
        uint32_t downloadMarkerArgs(
            DeviceEvaluationByteSlice* destination,
            uint32_t capacity) const;

        /// @brief Download the canonical firing-record index permutation.
        ///
        /// @param destination Caller-owned record-index output.
        /// @param capacity Destination element capacity.
        /// @return Number of canonical indices copied.
        /// @invariant `orderFiringRecords` completed after the last
        ///            materialization and `capacity` covers its result.
        uint32_t downloadFiringOrder(
            uint32_t* destination,
            uint32_t capacity) const;

        /// @brief Download the final packed doom line for every projected block.
        ///
        /// @param destination Caller-owned signed 64-bit doom-line output.
        /// @param capacity Destination element capacity.
        /// @return Number of logical-block doom lines copied.
        /// @invariant `selectDoomPrefixes` completed after the last firing sort
        ///            and `capacity` covers the uploaded logical-block count.
        uint32_t downloadDoomLines(
            int64_t* destination,
            uint32_t capacity) const;

        /// @brief Download the first triggering ordered-request index per block.
        ///
        /// @param destination Caller-owned unsigned request-index output.
        /// @param capacity Destination element capacity.
        /// @return Number of logical-block trigger indices copied.
        /// @invariant A doomed block carries the minimum request index at its
        ///            final packed doom line; an undoomed block carries UINT32_MAX.
        uint32_t downloadDoomRequestIndices(
            uint32_t* destination,
            uint32_t capacity) const;

        /// @brief Report all fixed evaluator bytes including shared CUB scratch.
        ///
        /// @return Exact startup bytes owned by this evaluator buffer.
        /// @invariant The value is constant for the object's lifetime.
        uint64_t fixedAllocationBytes() const;

    private:
        bool ownsDeviceAllocations_{ true };
        Phase2EvaluationCapacity capacity_{};
        void* deviceOutputBounds_{ nullptr };
        void* deviceStreamOwnerOffsets_{ nullptr };
        void* deviceStreamValueOffsets_{ nullptr };
        void* deviceStreamTotal_{ nullptr };
        uint64_t streamOwnerCount_{ 0 };
        uint64_t streamOwnerAt_{ 0 };
        uint64_t streamValueCount_{ 0 };
        uint64_t streamValueAt_{ 0 };
        uint32_t streamRequestCount_{ 0 };
        bool streamOpen_{ false };
        void* deviceRequestStates_{ nullptr };
        void* deviceRequestCounts_{ nullptr };
        void* deviceRequestOffsets_{ nullptr };
        void* deviceOwnerWork_{ nullptr };
        void* deviceCandidates_{ nullptr };
        void* deviceEncodedHits_{ nullptr };
        void* deviceValueCounts_{ nullptr };
        void* deviceValueOffsets_{ nullptr };
        void* deviceValueWork_{ nullptr };
        void* deviceFiringRecords_{ nullptr };
        void* deviceGeneratedBytes_{ nullptr };
        void* deviceLevelValues_{ nullptr };
        void* deviceOriginDependencies_{ nullptr };
        void* deviceMarkerKeys_{ nullptr };
        void* deviceMarkerRemainingArgs_{ nullptr };
        void* deviceMarkerArgs_{ nullptr };
        void* deviceFiringOrderA_{ nullptr };
        void* deviceFiringOrderB_{ nullptr };
        void* deviceFiringOrderResult_{ nullptr };
        void* deviceDoomLines_{ nullptr };
        void* deviceDoomRequestIndices_{ nullptr };
        void* deviceSelectedCount_{ nullptr };
        void* deviceCounters_{ nullptr };
        void* deviceTemporary_{ nullptr };
        std::size_t temporaryBytes_{ 0 };
        uint32_t usedCandidates_{ 0 };
        uint32_t usedEncodedHits_{ 0 };
        uint32_t usedValues_{ 0 };
        Phase2EvaluationStreamMetrics* streamMetrics_{nullptr};
        uint32_t usedFiringRecords_{ 0 };
        uint32_t usedGeneratedBytes_{ 0 };
        uint32_t usedLevelValues_{ 0 };
        uint32_t usedOriginDependencies_{ 0 };
        uint32_t usedMarkerKeys_{ 0 };
        uint32_t usedMarkerRemainingArgs_{ 0 };
        uint32_t usedMarkerArgs_{ 0 };
        uint32_t usedFiringOrder_{ 0 };
        uint32_t usedDoomLines_{ 0 };
    };

}  // namespace gl::gpu
