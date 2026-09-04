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

#include <cstddef>
#include <cstdint>

namespace gl::gpu {

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

    /// @brief Fixed ceilings for exact event ordering and request deduplication.
    ///
    /// @details
    /// Event capacity matches the persistent growth ledger, deduplication slots
    /// use a fixed power-of-two open-address table, and request capacity matches
    /// the retained raw-request ceiling.
    struct Phase2OrderingCapacity {
        uint32_t events{ 0 };
        uint32_t deduplicationSlots{ 0 };
        uint32_t requests{ 0 };
    };

    /// @brief Audited ordering capacity for the FTA shortcut route.
    inline constexpr Phase2OrderingCapacity
        kFtaShortcutPhase2OrderingCapacity{
            4194304, 524288, 262144 };

    /// @brief Audited ordering capacity for non-FTA full-run routes.
    inline constexpr Phase2OrderingCapacity kFullRunPhase2OrderingCapacity{
        4194304, 1048576, 524288 };

    /// @brief One exact-order unique request token for later device evaluation.
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

    /// @brief Host-visible result of exact ordering and semantic deduplication.
    struct Phase2OrderingResult {
        uint32_t orderedEventCount{ 0 };
        uint32_t uniqueRequestCount{ 0 };
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

    /// @brief Audited evaluation capacity for non-FTA full-run routes.
    inline constexpr Phase2EvaluationCapacity kFullRunPhase2EvaluationCapacity{
        kMaxProjectedBlocksPerChunk,
        524288,
        2097152,
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
    enum DevicePhase2FiringFlag : uint32_t {
        deviceFiringMarker = 1u << 0,
        deviceFiringOrdis2Demand = 1u << 1,
        deviceFiringProductOfDisintegration = 1u << 2,
        deviceFiringDisintegrationAllowed = 1u << 3,
        deviceFiringOrdisOnly = 1u << 4,
        deviceFiringSubsetExclusion = 1u << 5,
        deviceFiringDoNotDisintegrate = 1u << 6,
        deviceFiringAllowOrDisintegration = 1u << 7,
        deviceFiringAllGood = 1u << 8,
        deviceFiringAlreadyKnown = 1u << 9,
        deviceFiringMarkerNotAtomic = 1u << 10
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
    /// fixed generated-byte arena; rejected rows consume no output capacity.
    struct Phase2FiringExpressionResult {
        uint32_t firingRecordCount{ 0 };
        uint32_t generatedByteCount{ 0 };
        uint32_t levelValueCount{ 0 };
        uint32_t originDependencyCount{ 0 };
        uint32_t markerKeyCount{ 0 };
        uint32_t markerRemainingArgCount{ 0 };
        uint32_t markerArgCount{ 0 };
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
    static_assert(sizeof(Phase2OrderingCapacity) == 12);
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
        /// @brief Allocate all device projection storage once at engine startup.
        ///
        /// @details
        /// Validates the active CUDA device and allocates fixed arrays for logical
        /// blocks plus all 23 subordinate semantic columns and the checksum
        /// result. Every capacity is positive and every CUDA allocation asserts.
        ///
        /// @param fixedCapacity Immutable device element and byte ceilings.
        /// @return An empty device buffer owning all declared allocations.
        /// @invariant No method changes an allocation address or capacity.
        explicit CudaPhase2ProjectionBuffer(
            Phase2ProjectionCapacity fixedCapacity);

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

        /// @brief Join the evaluation-only projection upload stream.
        ///
        /// @details
        /// Synchronizes the process-owned nonblocking stream after independent
        /// host schedule construction has had an opportunity to overlap its
        /// copies. Every projection column is device-resident when the method
        /// returns, before the first semantic kernel starts.
        ///
        /// @return Nothing.
        /// @invariant Exactly one `beginPhase2Upload` is pending on entry.
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

    private:
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
        uint32_t usedCounts_[kProjectionColumnCount]{};
        uint64_t fixedAllocationBytes_{ 0 };
        bool phase2UploadPending_{ false };
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
        /// @brief Allocate every device task column once.
        ///
        /// @details
        /// Validates the CUDA device and positive capacities, then allocates fixed
        /// arrays for tasks, request batches, mandatory terms, and stumps plus one
        /// checksum word.
        ///
        /// @param fixedCapacity Immutable task-column element ceilings.
        /// @return An empty device task buffer owning all allocations.
        /// @invariant No method changes an allocation address or capacity.
        explicit CudaPhase2TaskBuffer(
            Phase2TaskProjectionCapacity fixedCapacity);

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
    /// classes; original calls map to shared compact spans before growth. A global
    /// radix sort orders class keys by class, decoded-name rank, and statement.
    class CudaPhase2FilterSortBuffer {
    public:
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
        /// @return An empty device filter/sort owner.
        /// @invariant No method changes an allocation address or capacity.
        explicit CudaPhase2FilterSortBuffer(
            Phase2FilterScheduleCapacity fixedCapacity);

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
        /// one stable compact class span; radix-sorts classes together; and maps
        /// original calls to their class count and offset. No processor semantic
        /// gate is executed on the host.
        ///
        /// @param projection Uploaded resident semantic projection.
        /// @param schedule Host schedule in processor call order.
        /// @return Total retained statement rows across all calls.
        /// @invariant Sorted keys encode class, decoded-name rank, and ascending
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

        /// @brief Download the most recent globally sorted composite keys.
        ///
        /// @details
        /// Copies the compact retained prefix. Each key is ordered and encoded as
        /// `(class, decoded-name rank, original statement index)` using the public
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
        /// @return An empty device growth owner.
        /// @invariant No method changes an allocation address or capacity.
        explicit CudaPhase2GrowthBuffer(Phase2GrowthCapacity fixedCapacity);

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
        friend class CudaPhase2OrderingBuffer;
        friend class CudaPhase2EvaluationBuffer;
        Phase2GrowthCapacity capacity_{};
        void* deviceCalls_{ nullptr };
        void* deviceViewMasks_{ nullptr };
        void* deviceSuffixMasks_{ nullptr };
        void* deviceFrontiers_[2]{};
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

    /// @brief Fixed-capacity CUDA ownership for exact event order and deduplication.
    ///
    /// @details
    /// Owns reusable radix-sort keys/indices, the preserved ordered event index,
    /// segmented growth-position scan arrays, a collision-exact request hash
    /// table, two request-token arrays, a count word, and one shared CUB scratch
    /// allocation. No ordering call allocates or falls back to the processor.
    class CudaPhase2OrderingBuffer {
    public:
        /// @brief Allocate all exact-order and deduplication scratch once.
        ///
        /// @details
        /// Validates positive ceilings and a power-of-two deduplication table,
        /// queries the maximum radix/scan temporary requirement, then allocates
        /// every fixed array and the shared CUB scratch region.
        ///
        /// @param fixedCapacity Immutable event, slot, and request ceilings.
        /// @return An empty reusable ordering owner.
        /// @invariant No later method changes an allocation address or capacity.
        explicit CudaPhase2OrderingBuffer(
            Phase2OrderingCapacity fixedCapacity);

        /// @brief Release every exact-order device allocation.
        ///
        /// @details
        /// Frees shared scratch, counters, request tokens, deduplication tables,
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

        /// @brief Reconstruct processor event order and deduplicate requests.
        ///
        /// @details
        /// Stable radix passes sort unordered events by task, call, stump run,
        /// and the exact stack-walk path token. A segmented inclusive scan assigns
        /// each event its task-local accepted-subkey tally. Recordable events then
        /// enter a full-key collision-checked per-call hash table that retains the
        /// minimum event order, and unique request tokens are radix-sorted by that
        /// order for later evaluation.
        ///
        /// @param projection Uploaded resident logical-block image.
        /// @param tasks Uploaded task and batch image used by the growth calls.
        /// @param growth Completed unordered growth content.
        /// @return Ordered event and unique request counts.
        /// @invariant Growth content and its uploaded call schedule are unchanged.
        Phase2OrderingResult orderAndDeduplicate(
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

        /// @brief Download unique request tokens in exact processor stream order.
        ///
        /// @param destination Caller-owned ordered-token output.
        /// @param capacity Destination element capacity.
        /// @return Number of unique request tokens copied.
        /// @invariant `capacity` covers the last unique request count.
        uint32_t downloadOrderedRequests(
            DeviceOrderedRequestToken* destination,
            uint32_t capacity) const;

        /// @brief Report every fixed ordering allocation including CUB scratch.
        ///
        /// @return Exact startup bytes owned by this ordering buffer.
        /// @invariant The value is constant for the object's lifetime.
        uint64_t fixedAllocationBytes() const;

    private:
        friend class CudaPhase2EvaluationBuffer;
        Phase2OrderingCapacity capacity_{};
        void* deviceKeys_[2]{};
        void* deviceIndices_[2]{};
        void* deviceOrderedIndices_{ nullptr };
        void* deviceScanValues_[2]{};
        void* deviceDeduplicationOwners_{ nullptr };
        void* deviceDeduplicationMinimumOrders_{ nullptr };
        void* deviceRequestTokens_[2]{};
        void* deviceUniqueRequestCount_{ nullptr };
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
        /// @brief Allocate every retained evaluator work and output array once.
        ///
        /// @details
        /// Validates all next-power-of-two ceilings, queries scan and selection
        /// temporary requirements, then allocates every work, firing, provenance,
        /// byte, ordering, doom, counter, and shared-scratch column. No evaluation
        /// call allocates.
        ///
        /// @param fixedCapacity Immutable evaluator work and output ceilings.
        /// @return An empty reusable evaluator owner.
        /// @invariant No later method changes an allocation address or capacity.
        explicit CudaPhase2EvaluationBuffer(
            Phase2EvaluationCapacity fixedCapacity);

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
        /// @param ordering Exact ordered unique request tokens.
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
        /// @param ordering Exact ordered unique request tokens.
        /// @return Exact firing-header and every variable-output used count.
        /// @invariant `expandEvaluationWork` was called for these same inputs and
        ///            rejected LMVs consume no output-arena capacity.
        Phase2FiringExpressionResult materializeFiringExpressions(
            const CudaPhase2ProjectionBuffer& projection,
            const CudaPhase2TaskBuffer& tasks,
            const CudaPhase2GrowthBuffer& growth,
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
        Phase2EvaluationCapacity capacity_{};
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
