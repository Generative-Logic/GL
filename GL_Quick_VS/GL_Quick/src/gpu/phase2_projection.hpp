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

#include "../memory_infra/int_encoded_expr.hpp"
#include "../parameters.hpp"

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace gl {
    struct ExpressionAnalyzer;
    struct ExpressionStump;
    struct Memory;
}

namespace gl::gpu {

    /// @brief Maximum projected logical blocks in one physical CUDA chunk.
    inline constexpr uint32_t kMaxProjectedBlocksPerChunk = 512;

    /// @brief Maximum task descriptors in one physical CUDA chunk.
    inline constexpr uint32_t kMaxPhase2TasksPerChunk = 1024;

    /// @brief Parallel host builders feeding one canonical projection chunk.
    inline constexpr std::size_t kPhase2ProjectionConstructionShardCount = 12;

    /// @brief Average complete-image shares reserved by each construction shard.
    inline constexpr uint32_t kPhase2ProjectionConstructionShardAverageShares = 2;

    /// @brief Divisor that gives each construction shard two average shares.
    inline constexpr uint32_t kPhase2ProjectionConstructionShardShareDivisor =
        static_cast<uint32_t>(kPhase2ProjectionConstructionShardCount)
        / kPhase2ProjectionConstructionShardAverageShares;

    /// @brief Named fixed-capacity image selected once per prover process.
    enum class Phase2ProjectionProfile : uint32_t {
        ftaShortcut = 0,
        fullRun = 1
    };

    /// @brief Fixed host projection ceilings reserved before logical blocks append.
    ///
    /// @details
    /// Each field is an element or byte ceiling for one reusable projection
    /// arena. The arena reserves every vector to these limits in its constructor,
    /// then asserts before each append; no logical-block append can grow storage.
    struct Phase2ProjectionCapacity {
        uint32_t logicalBlocks{ 0 };
        uint32_t statements{ 0 };
        uint32_t nameRecords{ 0 };
        uint32_t nameBytes{ 0 };
        uint32_t nameSlots{ 0 };
        uint32_t ruleStringRecords{ 0 };
        uint32_t ruleStringBytes{ 0 };
        uint32_t byteMapViews{ 0 };
        uint32_t byteMapEntries{ 0 };
        uint32_t byteMapSlots{ 0 };
        uint32_t byteKeyBytes{ 0 };
        uint32_t blobRecords{ 0 };
        uint32_t blobBytes{ 0 };
        uint32_t reverseMapViews{ 0 };
        uint32_t reverseMapEntries{ 0 };
        uint32_t reverseMapSlots{ 0 };
        uint32_t reverseKeyBytes{ 0 };
        uint32_t reverseOwners{ 0 };
        uint32_t podMapViews{ 0 };
        uint32_t podMapEntries{ 0 };
        uint32_t podMapSlots{ 0 };
        uint32_t podRunValues{ 0 };
        uint32_t mandatoryStatementKeys{ 0 };
        uint32_t metadataBytes{ 0 };
    };

    /// @brief Audited compact projection capacity for the FTA shortcut route.
    inline constexpr Phase2ProjectionCapacity
        kFtaShortcutPhase2ProjectionCapacity{
            kMaxProjectedBlocksPerChunk,
            262144,
            1048576,
            16777216,
            2097152,
            262144,
            16777216,
            5120,
            1048576,
            2097152,
            134217728,
            524288,
            33554432,
            512,
            131072,
            262144,
            16777216,
            131072,
            5632,
            1048576,
            4194304,
            1048576,
            262144,
            16384 };

    /// @brief Audited broad projection capacity for non-FTA full-run routes.
    inline constexpr Phase2ProjectionCapacity kFullRunPhase2ProjectionCapacity{
        kMaxProjectedBlocksPerChunk,
        131072,
        6540000,
        114000000,
        20600000,
        4990000,
        355000000,
        20000,
        11250000,
        32100000,
        2000000000,
        11800000,
        770000000,
        2048,
        1580000,
        4440000,
        301000000,
        1580000,
        22000,
        4990000,
        18700000,
        2290000,
        741000,
        91000 };

    /// @brief Return the audited fixed projection ceilings for one CUDA route.
    ///
    /// @details
    /// Centralizes the complete FTA-shortcut and full-run capacity profiles so
    /// host projection, device projection, and task-bearing block arrays share
    /// one named block ceiling. The result contains capacities only; it performs
    /// no allocation and reads no proof state.
    ///
    /// @param profile Fixed corpus profile selected for this process.
    /// @return Complete immutable capacity tuple for a projection arena.
    /// @invariant `logicalBlocks` always equals
    ///            `kMaxProjectedBlocksPerChunk`.
    Phase2ProjectionCapacity phase2ProjectionCapacityFor(
        Phase2ProjectionProfile profile);

    /// @brief Safe element and byte bounds for one complete Phase 2 read image.
    ///
    /// @details
    /// Counts the pointer-free columns required by request generation and request
    /// evaluation without allocating or serializing them. Byte-map figures cover
    /// all four normalized whole-key/subkey registries plus the overall encoded
    /// and remaining-argument maps. Pod-map figures cover statement membership,
    /// level runs, validity filters, recursion products, mail filters, and goals.
    /// Hash-slot counts use the projection's fixed maximum load of one half. The
    /// remaining-argument reverse map reports a no-dedup upper bound derived from
    /// its forward blobs; the builder may use a shorter prefix after coalescing
    /// duplicate normalized keys.
    struct Phase2ProjectionUsage {
        uint64_t logicalBlocks{ 0 };
        uint64_t statements{ 0 };
        uint64_t nameRecords{ 0 };
        uint64_t nameBytes{ 0 };
        uint64_t nameSlots{ 0 };
        uint64_t ruleStringRecords{ 0 };
        uint64_t ruleStringBytes{ 0 };
        uint64_t byteMapViews{ 0 };
        uint64_t byteMapEntries{ 0 };
        uint64_t byteMapSlots{ 0 };
        uint64_t byteKeyBytes{ 0 };
        uint64_t blobRecords{ 0 };
        uint64_t blobBytes{ 0 };
        uint64_t reverseMapViews{ 0 };
        uint64_t reverseMapEntries{ 0 };
        uint64_t reverseMapSlots{ 0 };
        uint64_t reverseKeyBytes{ 0 };
        uint64_t reverseOwners{ 0 };
        uint64_t podMapViews{ 0 };
        uint64_t podMapEntries{ 0 };
        uint64_t podMapSlots{ 0 };
        uint64_t podRunValues{ 0 };
        uint64_t mandatoryStatementKeys{ 0 };
        uint64_t metadataBytes{ 0 };
    };

    /// @brief Bound one resident logical block's selected Phase 2 projection.
    ///
    /// @details
    /// Walks only the exact read surfaces consumed by request generation and
    /// evaluation. The selection bit mask retains only the whole-key and subkey
    /// pair for each hash memory named by an executable request batch; unselected
    /// pairs retain zero-length descriptors at their fixed enum ordinals. The two
    /// overall-memory evaluation tables remain present because every batch can
    /// reach them after request growth. It counts canonical key/blob bytes from their resident cold
    /// records, fixed-load projection hash slots, interner bytes, membership
    /// rows, level/goal run values, mandatory-term statement keys, and scalar
    /// metadata. It performs no allocation, mutation, proof decision, or lookup
    /// substitution; callers may reduce the returned integers in any order.
    ///
    /// @param body Resident logical block whose post-Phase-1 read image is measured.
    /// @param analyzer Analyzer-wide immutable configuration and compiled-core
    ///                 registry used by Phase 2 evaluation.
    /// @param selectedHashMemories Bit `1 << DeviceHashMemoryKind` for every
    ///                 request memory consumed by this logical block's tasks.
    /// @return Exact direct-column usage plus safe derived-index upper bounds.
    /// @invariant The result is a pure function of the resident read image. It
    ///            may bound deterministic transfer chunks but never changes a
    ///            proof gate, request, or verdict.
    Phase2ProjectionUsage measurePhase2ProjectionUsage(
        const Memory& body,
        const ExpressionAnalyzer& analyzer,
        uint32_t selectedHashMemories = 0x0fu);

    /// @brief Compact compiled-core category retained by projected strings.
    ///
    /// @details
    /// Phase 2 needs only the exact distinction made by `allowedForMail` and
    /// marker staging: a missing compiled core, an atomic core, or any other
    /// compiled category. Projection computes this immutable annotation once;
    /// device evaluation never scans or duplicates the analyzer-wide map.
    enum class DeviceCompiledCategory : uint32_t {
        absent = 0,
        atomic = 1,
        nonAtomic = 2
    };

    /// @brief Pointer-free device record for one logical-block-local `NameMap` id.
    ///
    /// @details
    /// Byte offsets address the projection arena's packed name-byte array.
    /// `parentId` stays in the owning logical block's local identifier space.
    /// `decodedLexRank` is derived from the decoded bytes, never mint order, so
    /// device sorting reproduces host decoded-name order without string compares.
    struct DeviceNameRecord {
        uint32_t byteOffset{ 0 };
        uint32_t byteLength{ 0 };
        NameId parentId{ 0 };
        uint32_t decodedLexRank{ 0 };
        DeviceCompiledCategory compiledCategory{
            DeviceCompiledCategory::absent };
    };

    /// @brief Pointer-free byte slice for one rule-interner identifier.
    ///
    /// @details
    /// Identifier zero is an empty sentinel. Every other record points into the
    /// arena-wide rule-string byte column and preserves the owning logical
    /// block's `ValueInterner` identifier exactly.
    struct DeviceRuleStringRecord {
        uint32_t byteOffset{ 0 };
        uint32_t byteLength{ 0 };
        DeviceCompiledCategory compiledCategory{
            DeviceCompiledCategory::absent };
    };

    /// @brief Semantic identity of one projected byte-key lookup table.
    enum class DeviceByteMapKind : uint32_t {
        overallWholeKeys = 0,
        overallSubkeys = 1,
        localWholeKeys = 2,
        localSubkeys = 3,
        deltaWholeKeys = 4,
        deltaSubkeys = 5,
        workingWholeKeys = 6,
        workingSubkeys = 7,
        overallEncoded = 8,
        overallRemainingArgs = 9
    };

    /// @brief One byte-key table entry with an optional blob-record run.
    struct DeviceByteMapEntry {
        uint32_t keyOffset{ 0 };
        uint32_t keyLength{ 0 };
        uint32_t blobRecordOffset{ 0 };
        uint32_t blobRecordCount{ 0 };
    };

    /// @brief One serialized value record in the shared blob-byte column.
    struct DeviceBlobRecord {
        uint32_t byteOffset{ 0 };
        uint32_t byteLength{ 0 };
    };

    /// @brief Arena-relative descriptor for one fixed-load byte-key table.
    struct DeviceByteMapView {
        uint32_t entryOffset{ 0 };
        uint32_t entryCount{ 0 };
        uint32_t slotOffset{ 0 };
        uint32_t slotCount{ 0 };
        DeviceByteMapKind kind{ DeviceByteMapKind::overallWholeKeys };
    };

    /// @brief One derived normalized-key reverse-index entry.
    struct DeviceReverseMapEntry {
        uint32_t keyOffset{ 0 };
        uint32_t keyLength{ 0 };
        uint32_t ownerOffset{ 0 };
        uint32_t ownerCount{ 0 };
    };

    /// @brief Arena-relative descriptor for one remaining-argument reverse map.
    struct DeviceReverseMapView {
        uint32_t entryOffset{ 0 };
        uint32_t entryCount{ 0 };
        uint32_t slotOffset{ 0 };
        uint32_t slotCount{ 0 };
    };

    /// @brief Semantic identity of one projected plain-data lookup table.
    enum class DevicePodMapKind : uint32_t {
        recursionProducts = 0,
        knownStatements = 1,
        localStatements = 2,
        localDeltaStatements = 3,
        externalStatements = 4,
        statementLevels = 5,
        validityFilter = 6,
        frozenOrBranches = 7,
        goals = 8,
        mailEligibleStatements = 9,
        mailEligibleMarkers = 10
    };

    /// @brief Analyzer-wide bivalent switches copied into each resident block.
    enum DevicePhase2EvaluationFlag : uint32_t {
        deviceEvaluationIncubatorMode = 1u << 0,
        deviceEvaluationBanDisintegration = 1u << 1,
        deviceEvaluationCompressorMode = 1u << 2
    };

    /// @brief One widened plain-data key with scalar and run payloads.
    struct DevicePodMapEntry {
        int64_t key{ 0 };
        uint64_t scalar{ 0 };
        uint32_t runOffset{ 0 };
        uint32_t runCount{ 0 };
    };

    /// @brief Arena-relative descriptor for one fixed-load plain-data table.
    struct DevicePodMapView {
        uint32_t entryOffset{ 0 };
        uint32_t entryCount{ 0 };
        uint32_t slotOffset{ 0 };
        uint32_t slotCount{ 0 };
        DevicePodMapKind kind{ DevicePodMapKind::recursionProducts };
    };

    /// @brief Pointer-free slices locating one logical block in projection arrays.
    ///
    /// @details
    /// Every offset is relative to the corresponding arena-wide array. Name
    /// records include local identifier zero as a sentinel, so a local NameId can
    /// be added directly to `nameRecordOffset` after its range is checked.
    struct DeviceLogicalBlockProjection {
        uint32_t statementOffset{ 0 };
        uint32_t statementCount{ 0 };
        uint32_t nameRecordOffset{ 0 };
        uint32_t nameRecordCount{ 0 };
        uint32_t nameByteOffset{ 0 };
        uint32_t nameByteCount{ 0 };
        uint32_t nameSlotOffset{ 0 };
        uint32_t nameSlotCount{ 0 };
        uint32_t ruleStringRecordOffset{ 0 };
        uint32_t ruleStringRecordCount{ 0 };
        uint32_t ruleStringByteOffset{ 0 };
        uint32_t ruleStringByteCount{ 0 };
        uint32_t byteMapViewOffset{ 0 };
        uint32_t byteMapViewCount{ 0 };
        uint32_t reverseMapViewOffset{ 0 };
        uint32_t reverseMapViewCount{ 0 };
        uint32_t podMapViewOffset{ 0 };
        uint32_t podMapViewCount{ 0 };
        uint32_t localKeyOffset{ 0 };
        uint32_t localKeyCount{ 0 };
        uint32_t deltaKeyOffset{ 0 };
        uint32_t deltaKeyCount{ 0 };
        uint32_t externalKeyOffset{ 0 };
        uint32_t externalKeyCount{ 0 };
        uint32_t metadataOffset{ 0 };
        uint32_t metadataCount{ 0 };
        uint32_t anchorNameOffset{ 0 };
        uint32_t anchorNameCount{ 0 };
        int32_t overallMaxKeyLength{ 0 };
        int32_t localMaxKeyLength{ 0 };
        int32_t deltaMaxKeyLength{ 0 };
        int32_t workingMaxKeyLength{ 0 };
        int32_t primedForContradiction{ 0 };
        int32_t isPartOfRecursion{ 0 };
        int32_t contradictionIndex{ -1 };
        NameId mainValidityId{ 0 };
        int32_t level{ 0 };
        int32_t standardMaxSecondaryNumber{ 0 };
        uint32_t evaluationFlags{ 0 };
    };

    static_assert(std::is_trivially_copyable_v<DeviceNameRecord>);
    static_assert(std::is_trivially_copyable_v<DeviceRuleStringRecord>);
    static_assert(std::is_trivially_copyable_v<DeviceByteMapEntry>);
    static_assert(std::is_trivially_copyable_v<DeviceBlobRecord>);
    static_assert(std::is_trivially_copyable_v<DeviceByteMapView>);
    static_assert(std::is_trivially_copyable_v<DeviceReverseMapEntry>);
    static_assert(std::is_trivially_copyable_v<DeviceReverseMapView>);
    static_assert(std::is_trivially_copyable_v<DevicePodMapEntry>);
    static_assert(std::is_trivially_copyable_v<DevicePodMapView>);
    static_assert(std::is_trivially_copyable_v<DeviceLogicalBlockProjection>);
    static_assert(sizeof(DeviceNameRecord) == 20);
    static_assert(sizeof(DeviceRuleStringRecord) == 12);
    static_assert(sizeof(DeviceByteMapEntry) == 16);
    static_assert(sizeof(DeviceBlobRecord) == 8);
    static_assert(sizeof(DeviceByteMapView) == 20);
    static_assert(sizeof(DeviceReverseMapEntry) == 16);
    static_assert(sizeof(DeviceReverseMapView) == 16);
    static_assert(sizeof(DevicePodMapEntry) == 24);
    static_assert(sizeof(DevicePodMapView) == 20);
    static_assert(sizeof(DeviceLogicalBlockProjection) == 156);

    /// @brief Host-clock attribution for one cleared projection batch.
    ///
    /// @details
    /// Accumulates mutually exclusive nanosecond intervals inside
    /// `Phase2ProjectionArena::appendLogicalBlock`. The counters are diagnostic
    /// only: they never gate allocation, transfer, scheduling, or proof flow.
    /// `Phase2ProjectionArena::clear` resets the complete record before the next
    /// CUDA pass.
    struct Phase2ProjectionTiming {
        uint64_t logicalBlocks{ 0 };
        uint64_t preflightNanoseconds{ 0 };
        uint64_t statementNanoseconds{ 0 };
        uint64_t nameNanoseconds{ 0 };
        uint64_t nameRecordNanoseconds{ 0 };
        uint64_t nameSortNanoseconds{ 0 };
        uint64_t nameRankNanoseconds{ 0 };
        uint64_t nameSlotNanoseconds{ 0 };
        uint64_t ruleStringNanoseconds{ 0 };
        uint64_t byteMapNanoseconds{ 0 };
        uint64_t reverseMapNanoseconds{ 0 };
        uint64_t podMapNanoseconds{ 0 };
        uint64_t finalNanoseconds{ 0 };
        uint64_t mergeNanoseconds{ 0 };
    };

    /// @brief Reusable fixed-capacity host image for Phase 2 device projection.
    ///
    /// @details
    /// The constructor performs the only allocations by reserving all 24 public
    /// packed arrays and every private scratch array to its declared ceiling.
    /// `appendLogicalBlock` copies and indexes logical rows and bytes only;
    /// `clear` resets sizes while retaining all reservations for the next task
    /// batch.
    class Phase2ProjectionArena {
    public:
        Phase2ProjectionCapacity capacity;
        Phase2ProjectionTiming timing;
        std::vector<DeviceLogicalBlockProjection> logicalBlocks;
        std::vector<IntEncodedExpr> statements;
        std::vector<DeviceNameRecord> nameRecords;
        std::vector<char> nameBytes;
        std::vector<int32_t> nameSlots;
        std::vector<DeviceRuleStringRecord> ruleStringRecords;
        std::vector<char> ruleStringBytes;
        std::vector<DeviceByteMapView> byteMapViews;
        std::vector<DeviceByteMapEntry> byteMapEntries;
        std::vector<int32_t> byteMapSlots;
        std::vector<char> byteKeyBytes;
        std::vector<DeviceBlobRecord> blobRecords;
        std::vector<char> blobBytes;
        std::vector<DeviceReverseMapView> reverseMapViews;
        std::vector<DeviceReverseMapEntry> reverseMapEntries;
        std::vector<int32_t> reverseMapSlots;
        std::vector<char> reverseKeyBytes;
        std::vector<int32_t> reverseOwners;
        std::vector<DevicePodMapView> podMapViews;
        std::vector<DevicePodMapEntry> podMapEntries;
        std::vector<int32_t> podMapSlots;
        std::vector<int32_t> podRunValues;
        std::vector<int64_t> mandatoryStatementKeys;
        std::vector<char> metadataBytes;

        /// @brief Reserve every projection array to an immutable capacity ceiling.
        ///
        /// @details
        /// Asserts positive logical-block and name-record capacity, reserves all
        /// semantic arrays plus decoded-rank identifier, work, and range scratch
        /// once, and leaves every used length zero. Later appends assert before
        /// crossing any ceiling.
        ///
        /// @param fixedCapacity Element and byte ceilings for this arena lifetime.
        /// @return A reusable empty arena with all storage reserved.
        /// @invariant Vector capacities are at least their corresponding declared
        ///            ceilings for the arena's entire lifetime.
        explicit Phase2ProjectionArena(Phase2ProjectionCapacity fixedCapacity);

        /// @brief Reset all used prefixes without releasing reserved storage.
        ///
        /// @details
        /// Clears logical blocks, statements, names, bytes, and ranking scratch.
        /// It does not shrink or replace any vector, so the next deterministic task
        /// batch reuses the exact allocation set.
        ///
        /// @return Nothing.
        /// @invariant Every public used length is zero and every capacity remains
        ///            at least its constructor ceiling after return.
        void clear();

        /// @brief Append one resident logical block as pointer-free device slices.
        ///
        /// @details
        /// Copies statements, names, rule strings, selected hash-memory
        /// whole-key/subkey tables, fixed zero-length views for unselected hash
        /// memories, the overall evaluation tables, the
        /// derived remaining-argument reverse index, plain-data maps including
        /// the derived delta/external mandatory memberships, mandatory keys, and
        /// scalar metadata. It builds fixed-load lookup slots and
        /// decoded-name ranks using constructor-reserved scratch. Byte-identical
        /// name/id/parent tables and rule-string/id tables within one arena share
        /// their immutable packed slices after an exact row comparison; logical
        /// block descriptors remain distinct. All offsets are arena-relative.
        /// The operation asserts residency and every capacity boundary before a
        /// write; it never truncates, resizes, or skips a semantic row.
        ///
        /// @param body Resident logical block whose read-only Phase 2 state is read.
        /// @param analyzer Analyzer-wide immutable parameters, anchor name, and
        ///                 compiled-core categories read by evaluation.
        /// @param selectedHashMemories Bit `1 << DeviceHashMemoryKind` for every
        ///                 request memory consumed by this logical block's tasks.
        /// @return Pointer-free descriptor also appended to `logicalBlocks`.
        /// @invariant The appended slices decode byte-identically to `body` and
        ///            preserve statement index order and local identifier values.
        DeviceLogicalBlockProjection appendLogicalBlock(
            const Memory& body,
            const ExpressionAnalyzer& analyzer,
            uint32_t selectedHashMemories = 0x0fu);

        /// @brief Append one independently built projection shard without allocation.
        ///
        /// @details
        /// Copies all 24 used column prefixes from `source`, rebases every arena-wide
        /// descriptor, record, view, slot, byte, owner, and run offset onto this
        /// arena's existing prefixes, and accumulates the source construction timing.
        /// Source block order is retained inside the appended descriptor range; a
        /// caller may reorder those descriptors afterward without moving their slices.
        /// Empty shards are a defined no-op. Both arenas retain their constructor
        /// reservations and the source remains unchanged.
        ///
        /// @param source Independently built fixed-capacity shard to append.
        /// @return First logical-block index occupied by the appended shard.
        /// @invariant Every rebased reference addresses the byte-identical copied
        ///            source value and no destination vector reallocates.
        uint32_t appendShard(const Phase2ProjectionArena& source);

    private:
        /// @brief One pending byte-radix range over local name identifiers.
        ///
        /// @details
        /// `begin` and `end` address the decoded-rank identifier scratch, while
        /// `depth` selects the next packed-name byte. The explicit fixed-capacity
        /// range stack avoids recursion and live allocation during projection.
        struct DecodedRankRange {
            uint32_t begin{ 0 };
            uint32_t end{ 0 };
            uint32_t depth{ 0 };
        };

        /// @brief Exact read-only name projection already owned by this arena.
        ///
        /// @details
        /// The source remains resident and immutable throughout one CUDA chunk.
        /// A later logical block may reference these packed name columns only
        /// after byte-for-byte name-id and parent-id comparison succeeds.
        struct SharedNameProjection {
            const Memory* source{ nullptr };
            uint32_t recordOffset{ 0 };
            uint32_t recordCount{ 0 };
            uint32_t byteOffset{ 0 };
            uint32_t byteCount{ 0 };
            uint32_t slotOffset{ 0 };
            uint32_t slotCount{ 0 };
        };

        /// @brief Exact read-only rule-string projection already owned here.
        ///
        /// @details
        /// Rule ids are insertion-order ids. Sharing therefore requires equal
        /// counts and byte-identical values at every id before a descriptor may
        /// reuse the representative's record and byte slices.
        struct SharedRuleStringProjection {
            const Memory* source{ nullptr };
            uint32_t recordOffset{ 0 };
            uint32_t recordCount{ 0 };
            uint32_t byteOffset{ 0 };
            uint32_t byteCount{ 0 };
        };

        std::vector<NameId> decodedRankScratch_;
        std::vector<NameId> decodedRankWorkScratch_;
        std::vector<DecodedRankRange> decodedRankRangeScratch_;
        std::vector<uint32_t> reverseRunScratch_;
        std::vector<SharedNameProjection> sharedNameProjections_;
        std::vector<SharedRuleStringProjection> sharedRuleStringProjections_;
    };

    /// @brief Resident hash-memory identity selected by one request batch.
    enum class DeviceHashMemoryKind : uint32_t {
        overall = 0,
        local = 1,
        localDelta = 2,
        working = 3
    };

    /// @brief Semantic identity of one request batch inside a Phase 2 part.
    enum class DeviceRequestBatchKind : uint32_t {
        counterExample = 0,
        workingRules = 1,
        newThisBurst = 2,
        localRulesWithMail = 3,
        localDeltaRules = 4
    };

    /// @brief Resident statement-membership view named by a mandatory term.
    enum class DeviceMandatoryViewKind : uint32_t {
        local = 0,
        localDelta = 1,
        external = 2
    };

    /// @brief One pointer-free conjunction of mandatory statement views.
    struct DeviceMandatoryTerm {
        DeviceMandatoryViewKind views[2]{ DeviceMandatoryViewKind::local,
                                          DeviceMandatoryViewKind::local };
        uint32_t viewCount{ 0 };
    };

    /// @brief Host-side complete input for one projected request batch.
    struct Phase2RequestBatchInput {
        DeviceRequestBatchKind kind{ DeviceRequestBatchKind::counterExample };
        DeviceHashMemoryKind memory{ DeviceHashMemoryKind::overall };
        DeviceMandatoryTerm terms[2]{};
        uint32_t termCount{ 0 };
    };

    /// @brief Pointer-free device descriptor for one request batch.
    struct DeviceRequestBatch {
        DeviceRequestBatchKind kind{ DeviceRequestBatchKind::counterExample };
        DeviceHashMemoryKind memory{ DeviceHashMemoryKind::overall };
        uint32_t termOffset{ 0 };
        uint32_t termCount{ 0 };
    };

    /// @brief Pointer-free copy of one expression-split frontier stump.
    struct DeviceExpressionStump {
        NameId statementIndices[ExecutionParameters::MAX_EXPRESSIONS]{};
        NameId count{ 0 };
        uint32_t terminalOnly{ 0 };
    };

    /// @brief Pointer-free task descriptor for one Phase 2 executor part.
    struct DevicePhase2Task {
        uint32_t logicalBlockIndex{ 0 };
        uint32_t batchOffset{ 0 };
        uint32_t batchCount{ 0 };
        uint32_t stumpOffset{ 0 };
        uint32_t stumpCount{ 0 };
        NameId stumpOrdinal{ 0 };
        NameId stumpTotal{ 0 };
        int32_t maximumIterationNumberVariable{ 0 };
        uint32_t counterExampleMode{ 0 };
    };

    /// @brief Fixed ceilings for one reusable task-projection arena.
    struct Phase2TaskProjectionCapacity {
        uint32_t tasks{ 0 };
        uint32_t batches{ 0 };
        uint32_t terms{ 0 };
        uint32_t stumps{ 0 };
    };

    /// @brief Audited task-image capacity shared by both CUDA corpus profiles.
    inline constexpr Phase2TaskProjectionCapacity
        kPhase2TaskProjectionCapacity{
            kMaxPhase2TasksPerChunk, 4096, 4096, 16384 };

    /// @brief Exact fixed-column usage of one projected executor part.
    struct Phase2TaskProjectionUsage {
        uint64_t tasks{ 0 };
        uint64_t batches{ 0 };
        uint64_t terms{ 0 };
        uint64_t stumps{ 0 };
    };

    /// @brief Measure one executor part's complete task-projection shape.
    ///
    /// @details
    /// Reproduces the request-batch eligibility conditions at `performElem2`
    /// entry without generating requests or mutating proof state. Normal mode
    /// counts only batches with a satisfiable mandatory ingredient or a non-empty
    /// termless rule registry. A part with no eligible batch has zero task usage;
    /// counter-example mode always counts its single overall batch. The supplied
    /// stump count is copied only for a scheduled task.
    ///
    /// @param body Resident logical block whose batch inputs are inspected.
    /// @param stumpCount Number of stumps owned by this executor part.
    /// @param counterExampleMode Whether this is the unsplit counter-example route.
    /// @return Exact task, batch, term, and stump element counts; all zero when
    ///         normal mode has no structurally executable request batch.
    /// @invariant Observation-only; the result never gates processor proof flow.
    Phase2TaskProjectionUsage measurePhase2TaskProjectionUsage(
        const Memory& body,
        uint32_t stumpCount,
        bool counterExampleMode);

    static_assert(std::is_trivially_copyable_v<DeviceMandatoryTerm>);
    static_assert(std::is_trivially_copyable_v<Phase2RequestBatchInput>);
    static_assert(std::is_trivially_copyable_v<DeviceRequestBatch>);
    static_assert(std::is_trivially_copyable_v<DeviceExpressionStump>);
    static_assert(std::is_trivially_copyable_v<DevicePhase2Task>);
    static_assert(sizeof(DeviceMandatoryTerm) == 12);
    static_assert(sizeof(DeviceRequestBatch) == 16);
    static_assert(sizeof(DeviceExpressionStump) == 40);
    static_assert(sizeof(DevicePhase2Task) == 36);

    /// @brief Reusable fixed-capacity host image for Phase 2 part-local inputs.
    ///
    /// @details
    /// Separates executor-part state from the immutable resident logical-block
    /// projection. Batches identify their resident hash memory, terms identify
    /// resident mandatory-membership views, and split stumps retain statement
    /// indices and terminal markers exactly. Construction is the only allocation;
    /// append and clear never change vector capacities.
    class Phase2TaskProjectionArena {
    public:
        Phase2TaskProjectionCapacity capacity;
        std::vector<DevicePhase2Task> tasks;
        std::vector<DeviceRequestBatch> batches;
        std::vector<DeviceMandatoryTerm> terms;
        std::vector<DeviceExpressionStump> stumps;

        /// @brief Reserve every task-projection array to a fixed ceiling.
        ///
        /// @details
        /// Validates positive capacities, reserves all four arrays once, and
        /// leaves their used prefixes empty for the first task batch.
        ///
        /// @param fixedCapacity Immutable element ceilings for this arena lifetime.
        /// @return An empty task arena owning all host reservations.
        /// @invariant No later method changes an array's allocation capacity.
        explicit Phase2TaskProjectionArena(
            Phase2TaskProjectionCapacity fixedCapacity);

        /// @brief Reset task used prefixes while retaining fixed reservations.
        ///
        /// @details
        /// Clears tasks, batches, mandatory terms, and expression stumps without
        /// shrinking or replacing any vector.
        ///
        /// @return Nothing.
        /// @invariant Every used prefix is empty and every capacity is unchanged.
        void clear();

        /// @brief Append one executor part and all of its request-local inputs.
        ///
        /// @details
        /// Copies the ordered request-batch list and its ordered mandatory terms,
        /// then copies the split stump bucket exactly. An unsplit part has no
        /// stumps and zero ordinal/total; a split part has a non-empty stump run
        /// and a valid sibling position. All references become arena-relative
        /// offsets. Every boundary and bivalent marker asserts before write.
        ///
        /// @param logicalBlockIndex Index in the resident logical-block projection.
        /// @param batchInputs Ordered request batches executed by this part.
        /// @param batchCount Number of request batches; at least one.
        /// @param sourceStumps Optional contiguous processor stump bucket.
        /// @param stumpCount Number of source stumps; zero exactly when unsplit.
        /// @param stumpOrdinal Split sibling ordinal, or zero when unsplit.
        /// @param stumpTotal Split sibling count, or zero when unsplit.
        /// @param maximumIterationNumberVariable Request filter iteration ceiling.
        /// @param counterExampleMode Bivalent counter-example route marker.
        /// @return Pointer-free descriptor also appended to `tasks`.
        /// @invariant Output slices preserve batch, term, view, stump, and statement
        ///            order exactly and never reference processor addresses.
        DevicePhase2Task appendTask(
            uint32_t logicalBlockIndex,
            const Phase2RequestBatchInput* batchInputs,
            uint32_t batchCount,
            const ExpressionStump* sourceStumps,
            uint32_t stumpCount,
            NameId stumpOrdinal,
            NameId stumpTotal,
            int32_t maximumIterationNumberVariable,
            uint32_t counterExampleMode);
    };

    /// @brief One global filter/sort call over a resident logical block.
    ///
    /// @details
    /// Names the logical block, selected normalized-key registry, iteration
    /// ceiling, and the processor filter's whole-key widening flag. The resident
    /// logical-block descriptor supplies the statement slice and name ranks.
    struct DevicePhase2FilterCall {
        uint32_t logicalBlockIndex{ 0 };
        DeviceHashMemoryKind memory{ DeviceHashMemoryKind::overall };
        int32_t maximumIterationNumberVariable{ 0 };
        uint32_t alsoAcceptFullKeys{ 0 };
        uint32_t statementCount{ 0 };
    };

    /// @brief Fixed ceilings for one global statement filter/sort sweep.
    struct Phase2FilterScheduleCapacity {
        uint32_t calls{ 0 };
        uint32_t examinedRows{ 0 };
        uint32_t retainedRows{ 0 };
        uint32_t maximumExaminedRowsPerCall{ 0 };
    };

    /// @brief Audited filter schedule capacity shared by both corpus profiles.
    inline constexpr Phase2FilterScheduleCapacity
        kPhase2FilterScheduleCapacity{
            4096, 16777216, 1048576, 262144 };

    /// @brief Observation-only reuse census for one filter schedule.
    ///
    /// @details
    /// Counts exact filter-signature classes and the statement rows that one
    /// representative of each class would examine. Repeated calls and rows are
    /// reported separately so a later fixed-capacity class schedule can be sized
    /// from measured reuse without influencing the current processor-order route.
    struct Phase2FilterClassCensus {
        uint32_t callCount{ 0 };
        uint32_t uniqueClassCount{ 0 };
        uint32_t duplicateCallCount{ 0 };
        uint32_t uniqueExaminedRows{ 0 };
        uint32_t duplicateExaminedRows{ 0 };
        uint32_t maximumClassMultiplicity{ 0 };
    };

    inline constexpr uint32_t kDeviceFilterStatementIndexBits = 18;
    inline constexpr uint32_t kDeviceFilterNameRankBits = 18;
    inline constexpr uint32_t kDeviceFilterStatementIndexMask =
        (1u << kDeviceFilterStatementIndexBits) - 1u;
    inline constexpr uint32_t kDeviceFilterCallShift =
        kDeviceFilterStatementIndexBits + kDeviceFilterNameRankBits;

    static_assert(std::is_trivially_copyable_v<DevicePhase2FilterCall>);
    static_assert(sizeof(DevicePhase2FilterCall) == 20);

    /// @brief Reusable fixed-capacity host schedule for bulk statement filtering.
    ///
    /// @details
    /// Stores one pointer-free descriptor per request-generator or stump-producer
    /// filter call and interns exact semantic duplicates into one class. Appends
    /// preserve processor call order and map each original ordinal to an immutable
    /// class descriptor. Construction is the only allocation; clear retains every
    /// reservation and resets the fixed open-address table in place.
    class Phase2FilterScheduleArena {
    public:
        Phase2FilterScheduleCapacity capacity;
        std::vector<DevicePhase2FilterCall> calls;
        std::vector<DevicePhase2FilterCall> classes;
        std::vector<uint32_t> callClassIndices;
        std::vector<uint32_t> classMultiplicities;
        std::vector<int32_t> classSlots;
        uint32_t examinedRows{ 0 };
        uint32_t classExaminedRows{ 0 };

        /// @brief Reserve the global filter-call schedule once.
        ///
        /// @details
        /// Validates positive measured ceilings, reserves call, class, mapping,
        /// and multiplicity columns, allocates the
        /// fixed power-of-two class table, and leaves every used count zero.
        ///
        /// @param fixedCapacity Immutable call, examined-row, retained-row, and
        ///                      per-call ceilings.
        /// @return An empty reusable schedule.
        /// @invariant No later method changes any vector capacity.
        explicit Phase2FilterScheduleArena(
            Phase2FilterScheduleCapacity fixedCapacity);

        /// @brief Reset the global filter schedule without releasing storage.
        ///
        /// @details
        /// Clears call and class used prefixes, both examined-row accumulators,
        /// and every fixed class-table slot. The retained-row ceiling belongs to
        /// the paired CUDA result buffers.
        ///
        /// @return Nothing.
        /// @invariant Every allocation and declared ceiling is retained.
        void clear();

        /// @brief Append one processor-order filter call to the bulk schedule.
        ///
        /// @details
        /// Copies only scalar semantic selectors, interns the complete five-field
        /// signature through the fixed open-address table, and appends the class
        /// index beside the original processor-order call. Statement counts are
        /// accumulated for both original and unique-class work.
        ///
        /// @param logicalBlockIndex Resident logical-block projection index.
        /// @param memory Hash-memory registry supplying whole and subkey maps.
        /// @param maximumIterationNumberVariable Inclusive iteration ceiling.
        /// @param alsoAcceptFullKeys Bivalent processor whole-key widening flag.
        /// @param statementCount Full resident statement slice examined by the call.
        /// @return Pointer-free descriptor also appended to `calls`.
        /// @invariant Call order exactly matches the processor schedule; class
        ///            order is first occurrence; neither row total exceeds its
        ///            fixed ceiling.
        DevicePhase2FilterCall appendCall(
            uint32_t logicalBlockIndex,
            DeviceHashMemoryKind memory,
            int32_t maximumIterationNumberVariable,
            uint32_t alsoAcceptFullKeys,
            uint32_t statementCount);

        /// @brief Report exact filter-signature reuse in the production schedule.
        ///
        /// @details
        /// Reads the exact classes and multiplicities already built from logical
        /// block, hash-memory kind, iteration ceiling, whole-key widening flag,
        /// and statement count. The compact scan allocates no storage and never
        /// changes call order, row accounting, or a proof input.
        ///
        /// @return Exact class, duplicate-call, duplicate-row, and multiplicity
        ///         counts for the current used call prefix.
        /// @invariant The schedule and every proof-flow input remain unchanged.
        Phase2FilterClassCensus measureClassReuse() const;
    };

    /// @brief One request-generation call in the global device growth schedule.
    ///
    /// @details
    /// Links one projected executor task and one of its global batch records to
    /// the filter call that produced the candidate statement span. The three
    /// indices are independent arena-relative identifiers; no processor pointer
    /// crosses the host/device boundary.
    struct DevicePhase2GrowthCall {
        uint32_t taskIndex{ 0 };
        uint32_t batchIndex{ 0 };
        uint32_t filterCallIndex{ 0 };
    };

    /// @brief Reusable fixed-capacity host schedule for request frontier growth.
    ///
    /// @details
    /// Appends request calls in exact processor batch order after task and filter
    /// projection. Construction reserves the complete measured call ceiling;
    /// clear and append never replace the vector allocation.
    class Phase2GrowthScheduleArena {
    public:
        uint32_t capacity{ 0 };
        std::vector<DevicePhase2GrowthCall> calls;

        /// @brief Reserve the request-growth call schedule once.
        ///
        /// @details
        /// Requires a positive measured ceiling, reserves exactly that many
        /// pointer-free descriptors, and leaves the used prefix empty.
        ///
        /// @param fixedCapacity Immutable request-call ceiling.
        /// @return An empty reusable growth schedule.
        /// @invariant No later method changes the vector allocation capacity.
        explicit Phase2GrowthScheduleArena(uint32_t fixedCapacity);

        /// @brief Clear the request-growth schedule without releasing storage.
        ///
        /// @details
        /// Resets only the used prefix. The constructor reservation remains
        /// available for the next Phase 2 sweep.
        ///
        /// @return Nothing.
        /// @invariant The vector capacity remains at least `capacity`.
        void clear();

        /// @brief Append one processor-order request-growth call.
        ///
        /// @details
        /// Copies the task, global batch, and filter-call identifiers verbatim.
        /// Cross-column ownership is asserted when the paired CUDA buffers are
        /// launched, where their uploaded used lengths are available.
        ///
        /// @param taskIndex Global task descriptor index.
        /// @param batchIndex Global request-batch descriptor index.
        /// @param filterCallIndex Global filter-call descriptor index.
        /// @return Pointer-free descriptor also appended to `calls`.
        /// @invariant Append order is processor request-batch order.
        DevicePhase2GrowthCall appendCall(
            uint32_t taskIndex,
            uint32_t batchIndex,
            uint32_t filterCallIndex);
    };

    static_assert(std::is_trivially_copyable_v<DevicePhase2GrowthCall>);
    static_assert(sizeof(DevicePhase2GrowthCall) == 12);

    /// @brief Fixed element ceilings for device normalized-key growth scratch.
    ///
    /// @details
    /// `frontierRecords` applies independently to both ping-pong arrays and to the
    /// immutable per-wave prefix headers. The three prefix-value capacities own
    /// the normalized payload, distinct normalization-variable, and distinct
    /// secondary-variable pools measured across one live frontier.
    /// `candidateWindowRecords` bounds one deterministic attempt window and its
    /// compact survivor list independently of the much larger per-depth totals.
    /// Accepted events persist across expression depths so their exact processor
    /// order and growth positions can be reconstructed after breadth-parallel
    /// execution. Raw requests retain whole-key hits before semantic
    /// deduplication.
    struct Phase2GrowthCapacity {
        uint32_t calls{ 0 };
        uint32_t retainedRows{ 0 };
        uint32_t frontierRecords{ 0 };
        uint32_t acceptedEvents{ 0 };
        uint32_t rawRequests{ 0 };
        uint32_t prefixPayloadValues{ 0 };
        uint32_t prefixVariableValues{ 0 };
        uint32_t prefixSecondaryValues{ 0 };
        uint32_t candidateWindowRecords{ 0 };
    };

    /// @brief Audited growth capacity for the FTA shortcut route.
    inline constexpr Phase2GrowthCapacity kFtaShortcutPhase2GrowthCapacity{
        4096,
        1048576,
        4194304,
        4194304,
        262144,
        134217728,
        33554432,
        8388608,
        67108864 };

    /// @brief Audited growth capacity for non-FTA full-run routes.
    inline constexpr Phase2GrowthCapacity kFullRunPhase2GrowthCapacity{
        4096,
        1048576,
        4194304,
        4194304,
        524288,
        kFtaShortcutPhase2GrowthCapacity.prefixPayloadValues,
        kFtaShortcutPhase2GrowthCapacity.prefixVariableValues,
        kFtaShortcutPhase2GrowthCapacity.prefixSecondaryValues,
        16777216 };

    /// @brief Pointer-free node in one device normalized-key growth frontier.
    ///
    /// @details
    /// The candidate path stores positions in its call's already sorted filtered
    /// span, not statement copies or normalized-key bytes. The device reconstructs
    /// both from resident arrays. `runOrdinal` distinguishes stump searches inside
    /// one call; validity and mandatory-term masks are the resumable gate summary.
    struct DeviceGrowthNode {
        uint32_t callIndex{ 0 };
        uint32_t runOrdinal{ 0 };
        NameId filteredPositions[ExecutionParameters::MAX_EXPRESSIONS]{};
        NameId count{ 0 };
        NameId startPosition{ 0 };
        NameId validityId{ 0 };
        uint32_t termMask{ 0 };
    };

    /// @brief Immutable pooled request summary for one live growth node.
    ///
    /// @details
    /// Stores offsets and bounded lengths into the three fixed per-wave prefix
    /// pools. The packed summary contains hypothesis presence, the number of
    /// distinct non-exempt scopes, and both mandatory-term masks. Statement and
    /// filtered-position paths remain in `DeviceGrowthNode`; no semantic identity
    /// is duplicated here. Pool offsets may depend on thread arrival order, but
    /// they are never ordering keys and each header retains its exact slices.
    ///
    /// @invariant Every nonempty slice lies inside its matching fixed pool.
    struct DeviceGrowthPrefix {
        uint32_t payloadOffset{ 0 };
        uint32_t variableOffset{ 0 };
        uint32_t secondaryOffset{ 0 };
        uint16_t payloadLength{ 0 };
        uint16_t variableCount{ 0 };
        uint16_t secondaryCount{ 0 };
        uint16_t reserved{ 0 };
        NameId hypothesisValidity{ -1 };
        NameId nonExemptValidity{ -1 };
        uint32_t packedSummary{ 0 };
    };

    /// @brief Cheap-gate survivor inside one deterministic candidate window.
    ///
    /// @details
    /// Names one original frontier node and one position in its immutable sorted
    /// filter span. The cheap pass also carries the exact deeper validity and
    /// mandatory-view mask needed by suffix-only map probing and child emission.
    /// Window-local storage never becomes a semantic ordering identity; the node
    /// plus position reconstruct the canonical `(call, run, path, candidate)`
    /// ordinal.
    ///
    /// @invariant `nodeIndex` and `position` identify one in-capacity live
    ///            candidate that passed every gate through maximum key length.
    struct DeviceGrowthCandidateAttempt {
        uint32_t nodeIndex{ 0 };
        uint32_t position{ 0 };
        NameId validityId{ 0 };
        uint8_t termMask{ 0 };
        uint8_t reserved[3]{};
    };

    inline constexpr uint32_t kDeviceGrowthEventSubkeySatisfied = 1u << 0;
    inline constexpr uint32_t kDeviceGrowthEventWholeKeyPresent = 1u << 1;
    inline constexpr uint32_t kDeviceGrowthEventTermsSatisfied = 1u << 2;

    /// @brief Compact persistent record of one owner-accepted candidate event.
    ///
    /// @details
    /// Copies the logical-block-local statement path needed after its source
    /// frontier is reused. This also represents the stump-alone probe when a
    /// narrower registry filters out one of the producer's statements. The
    /// 16-bit position codes store filtered position plus one, leaving zero for a
    /// missing stump element; they are the compact exact-order input. Flags retain
    /// independent subkey, whole-key, and containment verdicts.
    struct DeviceAcceptedGrowthEvent {
        NameId statementIndices[ExecutionParameters::MAX_EXPRESSIONS]{};
        uint16_t filteredPositionCodes[
            ExecutionParameters::MAX_EXPRESSIONS]{};
        uint32_t callIndex{ 0 };
        uint32_t runOrdinal{ 0 };
        NameId count{ 0 };
        uint32_t flags{ 0 };
    };

    /// @brief One pre-dedup whole-key request with its accepted-event position.
    ///
    /// @details
    /// The embedded event carries the semantic candidate path. `growthPosition`
    /// is assigned only after accepted events are put into exact processor order;
    /// request deduplication retains the earliest such record per call.
    struct DeviceRawGrowthRequest {
        uint64_t growthPosition{ 0 };
        DeviceAcceptedGrowthEvent event{};
    };

    static_assert(std::is_trivially_copyable_v<Phase2GrowthCapacity>);
    static_assert(std::is_trivially_copyable_v<DeviceGrowthNode>);
    static_assert(std::is_trivially_copyable_v<DeviceGrowthPrefix>);
    static_assert(std::is_trivially_copyable_v<DeviceGrowthCandidateAttempt>);
    static_assert(std::is_trivially_copyable_v<DeviceAcceptedGrowthEvent>);
    static_assert(std::is_trivially_copyable_v<DeviceRawGrowthRequest>);
    static_assert(sizeof(Phase2GrowthCapacity) == 36);
    static_assert(sizeof(DeviceGrowthNode) == 56);
    static_assert(sizeof(DeviceGrowthPrefix) == 32);
    static_assert(sizeof(DeviceGrowthCandidateAttempt) == 16);
    static_assert(sizeof(DeviceAcceptedGrowthEvent) == 64);
    static_assert(sizeof(DeviceRawGrowthRequest) == 72);

}  // namespace gl::gpu
