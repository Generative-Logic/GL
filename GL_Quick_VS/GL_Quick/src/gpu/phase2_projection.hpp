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

#include <cassert>

#include "../memory_infra/int_encoded_expr.hpp"
#include "../parameters.hpp"
#include "phase2_usage.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace gl {
    struct ExpressionAnalyzer;
    struct ExpressionStump;
    struct Memory;
}

namespace gl::gpu {

    struct Phase2ProjectionKeyStore;

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

    /// @brief Field-wise sum of two projection capacities.
    ///
    /// @details
    /// Lays construction-shard capacities end to end: the storage of a backing
    /// arena that holds several shards side by side, and each shard's slice
    /// start, are running sums of the shards' capacities
    /// (D-425). A sum that leaves `uint32_t` asserts.
    ///
    /// @param left First capacity.
    /// @param right Second capacity.
    /// @return The field-wise sum.
    Phase2ProjectionCapacity addProjectionCapacity(const Phase2ProjectionCapacity& left,
        const Phase2ProjectionCapacity& right);

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

    /// @brief Partition construction work into waves safe for any worker assignment.
    ///
    /// @details
    /// Each consecutive wave's unshared usage fits one construction shard in
    /// every column. A worker may therefore claim the whole wave without
    /// exceeding its reservation. Device chunks and logical-block order remain
    /// unchanged; completed waves append to the same final projection.
    ///
    /// @param usages Per-block unshared projection bounds in canonical order.
    /// @param count Number of blocks; zero describes an empty construction job.
    /// @param capacity Fixed capacity of one construction shard.
    /// @param ends Output exclusive end indices of consecutive waves.
    /// @param endCapacity Number of available output entries, at least count.
    /// @return Number of nonempty waves, or zero for an empty job.
    /// @invariant Every individual block and every emitted wave fits the shard.
    /// @see measurePhase2ProjectionUsage, Phase2ProjectionArena::adoptShards.
    uint32_t planProjectionConstructionWaves(const Phase2ProjectionUsage* usages,
        uint32_t count, const Phase2ProjectionCapacity& capacity,
        uint32_t* ends, uint32_t endCapacity);

    /// @brief Measure shared key bytes and exact next-predicate table storage.
    ///
    /// @details Counts the byte union of the selected source maps and their
    /// distinct immediate prefixes in private arena scratch. Prefix membership
    /// and predicate transitions stay separate for each hash-memory kind. The
    /// source stores are read-only and no proof identifier is minted.
    ///
    /// @param body Claimed resident logical block whose source maps are stable.
    /// @param selectedHashMemories Nonzero four-bit request-memory selection.
    /// @param buildPrefixFilter Include the four next-predicate table shapes.
    /// @return Shared byteKeyBytes and the prefix-only entry, slot, blob-record
    ///         and blob-byte counts; every other usage field is zero.
    /// @invariant Equal bytes may share storage but never merge map membership.
    /// @see measurePhase2ProjectionUsage, Phase2ProjectionArena::appendLogicalBlock
    Phase2ProjectionUsage measurePhase2SharedKeyUsage(const Memory& body,
        uint32_t selectedHashMemories, bool buildPrefixFilter);

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
    /// metadata. It uses private arena scratch without source mutation, proof decisions, or lookup
    /// substitution; callers may reduce the returned integers in any order.
    ///
    /// @param body Resident logical block whose post-Phase-1 read image is measured.
    /// @param analyzer Analyzer-wide immutable configuration and compiled-core
    ///                 registry used by Phase 2 evaluation.
    /// @param selectedHashMemories Bit `1 << DeviceHashMemoryKind` for every
    ///                 request memory consumed by this logical block's tasks.
    /// @param buildPrefixFilter Include derived next-predicate maps used by the
    ///                 production grower; false retains the projection test oracle.
    /// @return Exact direct-column usage plus safe derived-index upper bounds.
    /// @invariant The result is a pure function of the resident read image. It
    ///            may bound deterministic transfer chunks but never changes a
    ///            proof gate, request, or verdict.
    Phase2ProjectionUsage measurePhase2ProjectionUsage(
        const Memory& body,
        const ExpressionAnalyzer& analyzer,
        uint32_t selectedHashMemories = 0x0fu,
        bool buildPrefixFilter = false);

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
        overallRemainingArgs = 9,
        overallNextPredicates = 10,
        localNextPredicates = 11,
        deltaNextPredicates = 12,
        workingNextPredicates = 13
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
    /// @details keyOffset addresses immutable forward blobBytes; ownerOffset
    /// addresses the separate reverseOwners column.
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
        goals = 8
        // 9 and 10 are retired mail-permission projection kinds.
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
        int32_t recursionAuxyImplId{ 0 };
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
    static_assert(sizeof(DeviceLogicalBlockProjection) == 160);

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
        uint64_t sourceStatementRows{ 0 };
        uint64_t retainedStatementRows{ 0 };
        uint64_t preflightNanoseconds{ 0 };
        uint64_t statementNanoseconds{ 0 };
        uint64_t nameNanoseconds{ 0 };
        uint64_t nameRecordNanoseconds{ 0 };
        uint64_t nameSortNanoseconds{ 0 };
        uint64_t nameRankNanoseconds{ 0 };
        uint64_t nameSlotNanoseconds{ 0 };
        uint64_t ruleStringNanoseconds{ 0 };
        uint64_t byteMapNanoseconds{ 0 };
        uint64_t prefixFilterNanoseconds{ 0 };
        uint64_t prefixFilterKeys{ 0 }, prefixFilterPredicates{ 0 }, prefixFilterBytes{ 0 };
        uint64_t reverseMapNanoseconds{ 0 };
        uint64_t podMapNanoseconds{ 0 };
        uint64_t finalNanoseconds{ 0 };
        uint64_t mergeNanoseconds{ 0 };
    };

    /// @brief Heap bytes reserved by a set of `std::vector` columns — the
    ///        host-memory measure of the fixed-capacity CUDA host arenas.
    ///
    /// @details
    /// Every CUDA host arena reserves its columns once to a fixed capacity at
    /// construction, so `capacity() * sizeof(value_type)` summed over the
    /// columns is exactly the heap the arena pins while it lives. Used by the
    /// arenas' `hostBytes()` accessors for the memory playbook's per-chunk
    /// census. Observability only (Rule 16).
    ///
    /// @tparam Vectors `std::vector` column types.
    /// @param vectors The columns to measure.
    /// @return Sum of reserved bytes; 0 for no columns.
    template <class... Vectors>
    inline uint64_t vectorCapacityBytes(const Vectors&... vectors) {
        return (static_cast<uint64_t>(0) + ... + static_cast<uint64_t>(
            vectors.capacity() * sizeof(typename Vectors::value_type)));
    }

    /// @brief Count owned host column bytes for measured projection capacities.
    ///
    /// @details Includes source statement ordinals and all device columns, but
    /// excludes construction scratch. The bounded 32-bit counts and fixed type
    /// widths make the weighted sum representable in 64 bits.
    /// @param c Per-column measured capacities.
    /// @return Required host column bytes.
    inline uint64_t phase2ProjectionHostBytes(const Phase2ProjectionCapacity& c) {
        return
            uint64_t(c.logicalBlocks) * (sizeof(DeviceLogicalBlockProjection))
            + uint64_t(c.statements) * (sizeof(IntEncodedExpr) + sizeof(NameId))
            + uint64_t(c.nameRecords) * (sizeof(DeviceNameRecord))
            + uint64_t(c.nameBytes) * (1)
            + uint64_t(c.nameSlots) * (sizeof(int32_t))
            + uint64_t(c.ruleStringRecords) * (sizeof(DeviceRuleStringRecord))
            + uint64_t(c.ruleStringBytes) * (1)
            + uint64_t(c.byteMapViews) * (sizeof(DeviceByteMapView))
            + uint64_t(c.byteMapEntries) * (sizeof(DeviceByteMapEntry))
            + uint64_t(c.byteMapSlots) * (sizeof(int32_t))
            + uint64_t(c.byteKeyBytes) * (1)
            + uint64_t(c.blobRecords) * (sizeof(DeviceBlobRecord))
            + uint64_t(c.blobBytes) * (1)
            + uint64_t(c.reverseMapViews) * (sizeof(DeviceReverseMapView))
            + uint64_t(c.reverseMapEntries) * (sizeof(DeviceReverseMapEntry))
            + uint64_t(c.reverseMapSlots) * (sizeof(int32_t))
            + uint64_t(c.reverseKeyBytes) * (1)
            + uint64_t(c.reverseOwners) * (sizeof(int32_t))
            + uint64_t(c.podMapViews) * (sizeof(DevicePodMapView))
            + uint64_t(c.podMapEntries) * (sizeof(DevicePodMapEntry))
            + uint64_t(c.podMapSlots) * (sizeof(int32_t))
            + uint64_t(c.podRunValues) * (sizeof(int32_t))
            + uint64_t(c.mandatoryStatementKeys) * (sizeof(int64_t))
            + uint64_t(c.metadataBytes) * (1);
    }

    /// @brief Fixed-capacity column of one projection record type that owns its
    ///        storage or views a slice of another column's storage.
    ///
    /// @details
    /// The part of the `std::vector` interface the projection uses (size, data,
    /// indexing, iteration, appending at the end, fill-resizing, clearing) over
    /// a capacity fixed once. `reserve` allocates and owns exactly that many
    /// elements. `attachSlice` instead views part of a larger owning column, so
    /// construction shards write straight into their slice of the final host
    /// projection, and `appendMoved` later compacts a slice's used prefix onto
    /// the owner's end inside the same storage
    /// (D-425). No operation reallocates: growth past
    /// the capacity is an assert, because each construction window sizes columns from
    /// measured usage.
    ///
    /// @tparam T Trivially copyable element type.
    /// @invariant `size() <= capacity()`; a slice never outlives its owner.
    template <class T>
    class ProjectionColumn {
        static_assert(std::is_trivially_copyable_v<T>);

    public:
        using value_type = T;

        ProjectionColumn() = default;

        /// @brief Free owned storage.
        ///
        /// @details
        /// A slice frees nothing: its storage belongs to the owning column.
        ~ProjectionColumn() {
            if (owned_) std::allocator<T>().deallocate(data_, capacity_);
        }

        ProjectionColumn(const ProjectionColumn&) = delete;
        ProjectionColumn& operator=(const ProjectionColumn&) = delete;

        /// @brief Allocate and own storage for exactly `capacity` elements.
        ///
        /// @details
        /// The column must hold no storage yet. The storage stays until the
        /// column is destroyed; `size()` starts at zero.
        ///
        /// @param capacity Element count; positive.
        /// @return Nothing.
        void reserve(std::size_t capacity) {
            assert(data_ == nullptr && capacity > 0);
            data_ = std::allocator<T>().allocate(capacity);
            capacity_ = capacity;
            owned_ = true;
        }

        /// @brief View `capacity` elements of `storage` starting at `offset`.
        ///
        /// @details
        /// The column must hold no storage yet and `storage` must own its
        /// storage and cover the whole range. Appends write straight into
        /// `storage`'s elements; `storage.size()` is not changed.
        ///
        /// @param storage Owning column; must outlive this slice.
        /// @param offset First viewed element.
        /// @param capacity Viewed element count.
        /// @return Nothing.
        void attachSlice(ProjectionColumn& storage, std::size_t offset,
                         std::size_t capacity) {
            assert(data_ == nullptr && storage.owned_);
            assert(offset <= storage.capacity_ && capacity <= storage.capacity_ - offset);
            data_ = storage.data_ + offset;
            capacity_ = capacity;
        }

        /// @brief Elements in use.
        ///
        /// @return The used length.
        std::size_t size() const { return size_; }

        /// @brief Elements the column can hold.
        ///
        /// @return The fixed capacity (the slice length for a slice).
        std::size_t capacity() const { return capacity_; }

        /// @brief Whether no element is in use.
        ///
        /// @return `size() == 0`.
        bool empty() const { return size_ == 0; }

        /// @brief Heap bytes this column owns.
        ///
        /// @return `capacity() * sizeof(T)` when owning, zero for a slice or an
        ///         unreserved column.
        uint64_t ownedBytes() const {
            return owned_ ? static_cast<uint64_t>(capacity_) * sizeof(T) : 0;
        }

        /// @brief First element.
        ///
        /// @return Pointer to the storage (null before `reserve` / `attachSlice`).
        T* data() { return data_; }

        /// @brief First element, read-only.
        ///
        /// @return Pointer to the storage (null before `reserve` / `attachSlice`).
        const T* data() const { return data_; }

        /// @brief Element `index`; unchecked like `std::vector`.
        ///
        /// @param index Position below `size()`.
        /// @return The element.
        T& operator[](std::size_t index) { return data_[index]; }

        /// @brief Element `index`, read-only; unchecked like `std::vector`.
        ///
        /// @param index Position below `size()`.
        /// @return The element.
        const T& operator[](std::size_t index) const { return data_[index]; }

        /// @brief Iterator to the first element.
        ///
        /// @return `data()`.
        T* begin() { return data_; }

        /// @brief Iterator past the last used element.
        ///
        /// @return `data() + size()`.
        T* end() { return data_ + size_; }

        /// @brief Read-only iterator to the first element.
        ///
        /// @return `data()`.
        const T* begin() const { return data_; }

        /// @brief Read-only iterator past the last used element.
        ///
        /// @return `data() + size()`.
        const T* end() const { return data_ + size_; }

        /// @brief Last used element; the column must not be empty.
        ///
        /// @return The element at `size() - 1`.
        T& back() {
            assert(size_ > 0);
            return data_[size_ - 1];
        }

        /// @brief Append one element; the column must not be full.
        ///
        /// @param value Element to append.
        /// @return Nothing.
        void push_back(const T& value) {
            assert(size_ < capacity_);
            data_[size_++] = value;
        }

        /// @brief Set the used length, filling new positions with `value`.
        ///
        /// @details
        /// Shrinking only lowers the used length; growing past the capacity is
        /// an assert.
        ///
        /// @param count New used length.
        /// @param value Value of the new elements (value-initialized by default).
        /// @return Nothing.
        void resize(std::size_t count, const T& value = T{}) {
            assert(count <= capacity_);
            for (std::size_t index = size_; index < count; ++index) data_[index] = value;
            size_ = count;
        }

        /// @brief Append the range `[first, last)`; `position` must be `end()`.
        ///
        /// @tparam Iterator Input iterator over values convertible to `T`.
        /// @param position Insertion point; only the end is supported.
        /// @param first First value.
        /// @param last Past the last value.
        /// @return Nothing.
        template <class Iterator>
        void insert(const T* position, Iterator first, Iterator last) {
            assert(position == data_ + size_);
            const auto count = static_cast<std::size_t>(std::distance(first, last));
            assert(count <= capacity_ - size_);
            std::copy(first, last, data_ + size_);
            size_ += count;
        }

        /// @brief Set the used length to zero, keeping the storage.
        ///
        /// @return Nothing.
        void clear() { size_ = 0; }

        /// @brief Move a slice's used elements to this owning column's end.
        ///
        /// @details
        /// `slice` must view this column's storage at or after `end()`, so the
        /// move is a `memmove` towards the front of the same storage and never
        /// overwrites a later slice (slices are moved in address order). A slice
        /// already in place moves nothing. `slice` ends empty.
        ///
        /// @param slice A slice of this column.
        /// @return Nothing.
        void appendMoved(ProjectionColumn& slice) {
            T* const destination = data_ + size_;
            assert(owned_ && !slice.owned_);
            assert(slice.data_ >= destination
                && slice.data_ + slice.size_ <= data_ + capacity_);
            if (slice.size_ > 0 && slice.data_ != destination)
                std::memmove(destination, slice.data_, slice.size_ * sizeof(T));
            size_ += slice.size_;
            slice.size_ = 0;
        }

        /// @brief Equal used lengths and element-wise equal values.
        ///
        /// @param left First column.
        /// @param right Second column.
        /// @return Whether both columns hold the same used elements.
        friend bool operator==(const ProjectionColumn& left, const ProjectionColumn& right) {
            return left.size_ == right.size_
                && std::equal(left.begin(), left.end(), right.begin());
        }

    private:
        T* data_ = nullptr;
        std::size_t size_ = 0;
        std::size_t capacity_ = 0;
        bool owned_ = false;
    };

    /// @brief Persistent host threads that run the jobs of one projection chunk.
    ///
    /// @details
    /// Owns a fixed number of threads for its whole lifetime and runs one job at
    /// a time. `run` offers every task index in `[0, taskCount)` to the threads,
    /// which claim indices from one shared counter, and returns only after every
    /// task has finished, so the caller observes all writes of the job. The CUDA
    /// projection runs each construction window and each shard adoption of a
    /// chunk on one set of workers instead of starting threads per window
    /// (D-447). The workers hold no
    /// logical-block state; every job's data lives with its caller. Which thread
    /// runs a task is unobservable: tasks of one job must be independent.
    class Phase2ProjectionWorkers {
    public:
        /// @brief Start the persistent threads.
        ///
        /// @details
        /// The threads wait for the first job. Nothing else is allocated later:
        /// jobs reuse the same threads, mutex and condition variables.
        ///
        /// @param threadCount Number of threads; positive.
        /// @return Idle workers.
        explicit Phase2ProjectionWorkers(uint32_t threadCount);

        /// @brief Stop and join every thread.
        ///
        /// @details
        /// Runs only between jobs (`run` returns after its job), so no task is
        /// in flight; each thread observes the stop request and exits.
        ///
        /// @return Nothing.
        ~Phase2ProjectionWorkers();

        Phase2ProjectionWorkers(const Phase2ProjectionWorkers&) = delete;
        Phase2ProjectionWorkers& operator=(const Phase2ProjectionWorkers&) = delete;

        /// @brief Run `job(task)` for every task index in `[0, taskCount)`.
        ///
        /// @details
        /// The calling thread waits while the persistent threads claim indices
        /// from a shared counter; each index runs exactly once. The call returns
        /// after the last task has finished, with the job's writes visible to the
        /// caller. One job runs at a time; the calling thread must not be one of
        /// the workers.
        ///
        /// @param taskCount Number of task indices; zero returns at once.
        /// @param job Callable taking one `uint32_t` task index; it outlives the call.
        /// @return Nothing.
        template <class Job>
        void run(uint32_t taskCount, Job& job) {
            dispatch(taskCount, &invokeJob<Job>, &job);
        }

    private:
        /// @brief Call a typed job through the type-erased dispatch.
        ///
        /// @param job The caller's job object.
        /// @param task Task index to run.
        /// @return Nothing.
        template <class Job>
        static void invokeJob(void* job, uint32_t task) {
            (*static_cast<Job*>(job))(task);
        }

        /// @brief Publish one job to the threads and wait for its last task.
        ///
        /// @details
        /// Under the mutex stores the job, resets the task counter and the count
        /// of threads still working, advances the generation that wakes every
        /// thread, then waits until each thread has finished claiming tasks.
        ///
        /// @param taskCount Number of task indices.
        /// @param invoke Type-erased call of the job.
        /// @param job The caller's job object.
        /// @return Nothing.
        /// @invariant No other job is in flight and the workers are not stopping.
        void dispatch(uint32_t taskCount, void (*invoke)(void*, uint32_t), void* job);

        /// @brief Thread body: wait for a new generation, claim its tasks, report.
        ///
        /// @details
        /// Each thread runs claimed tasks until the counter passes `taskCount`,
        /// then decrements the count of working threads; the last one wakes the
        /// dispatcher. A stop request with no new generation ends the thread.
        ///
        /// @return Nothing.
        void work();

        std::mutex mutex_;
        std::condition_variable start_;
        std::condition_variable finished_;
        std::atomic<uint32_t> nextTask_{0};
        void (*invoke_)(void*, uint32_t) = nullptr;
        void* job_ = nullptr;
        uint32_t taskCount_ = 0;
        uint32_t workingThreads_ = 0;
        uint64_t generation_ = 0;
        bool stopping_ = false;
        std::vector<std::thread> threads_;
    };

    /// @brief Reusable fixed-capacity host image for Phase 2 device projection.
    ///
    /// @details
    /// The constructors perform the only allocations. An owning arena reserves
    /// all 25 public packed columns and every private scratch array to its
    /// declared ceiling. The production route builds each chunk in consecutive
    /// host windows: a backing arena reserves one window's construction shards
    /// laid end to end, each shard is a slice arena whose columns view its part
    /// of that storage, and
    /// `adoptShards` compacts the shards' rows in place. `appendLogicalBlock`
    /// copies and indexes logical rows and bytes only. Each complete window is
    /// rebased and uploaded before its storage is released; a sealing arena
    /// retains only descriptors, lookup data and source ordinals. `clear` resets sizes
    /// while retaining all reservations for the next task batch.
    class Phase2ProjectionArena {
    public:
        Phase2ProjectionCapacity capacity;
        Phase2ProjectionTiming timing;
        ProjectionColumn<DeviceLogicalBlockProjection> logicalBlocks;
        ProjectionColumn<IntEncodedExpr> statements;
        // Host-only source ordinals, parallel to device statements; retained for scheduling.
        ProjectionColumn<NameId> sourceStatementIndices;
        ProjectionColumn<DeviceNameRecord> nameRecords;
        ProjectionColumn<char> nameBytes;
        ProjectionColumn<int32_t> nameSlots;
        ProjectionColumn<DeviceRuleStringRecord> ruleStringRecords;
        ProjectionColumn<char> ruleStringBytes;
        ProjectionColumn<DeviceByteMapView> byteMapViews;
        ProjectionColumn<DeviceByteMapEntry> byteMapEntries;
        ProjectionColumn<int32_t> byteMapSlots;
        ProjectionColumn<char> byteKeyBytes;
        ProjectionColumn<DeviceBlobRecord> blobRecords;
        ProjectionColumn<char> blobBytes;
        ProjectionColumn<DeviceReverseMapView> reverseMapViews;
        ProjectionColumn<DeviceReverseMapEntry> reverseMapEntries;
        ProjectionColumn<int32_t> reverseMapSlots;
        // Empty transport column: reverse keys address the forward blobBytes.
        ProjectionColumn<char> reverseKeyBytes;
        ProjectionColumn<int32_t> reverseOwners;
        ProjectionColumn<DevicePodMapView> podMapViews;
        ProjectionColumn<DevicePodMapEntry> podMapEntries;
        ProjectionColumn<int32_t> podMapSlots;
        ProjectionColumn<int32_t> podRunValues;
        ProjectionColumn<int64_t> mandatoryStatementKeys;
        ProjectionColumn<char> metadataBytes;
        // Host-only, sealing image: the rule copy (`packRuleOwner` of the LMV's
        // original implication and scope) of each complete-image blob record
        // of an overall encoded-map run, zero for any other blob record; read
        // by the sealing of marker records (I-284).
        ProjectionColumn<int64_t> blobRuleOwners;

        /// @brief Select a retained host image containing only sealing inputs.
        struct SealingOnly {};

        /// @brief Reserve only sealing lookups and source statement ordinals.
        ///
        /// @details Device-only columns own no storage. Uploaded windows append
        /// rebased retained data; this arena never constructs logical blocks.
        /// @param fixedCapacity Complete chunk limits for retained columns.
        /// @param mode Explicit retained-only storage selection.
        /// @return An empty retained host image.
        Phase2ProjectionArena(Phase2ProjectionCapacity fixedCapacity, SealingOnly mode);

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

        /// @brief Reserve the columns of a projection whose construction shards
        ///        write straight into it.
        ///
        /// @details
        /// The backing arena of a build-once projection
        /// (D-425): every column reserves `storage`
        /// elements (the shards' capacities laid end to end, see
        /// `addProjectionCapacity`) while `capacity` stays the ceiling the
        /// adopted totals must fit, the size the device buffer is built for.
        /// Reserves no construction scratch: the arena never appends logical
        /// blocks itself (asserted) and receives its rows once, through
        /// `adoptShards`.
        ///
        /// @param fixedCapacity Ceilings of the adopted totals; every field positive.
        /// @param storage Elements each column reserves; every field positive.
        /// @return An empty backing arena.
        Phase2ProjectionArena(Phase2ProjectionCapacity fixedCapacity,
            const Phase2ProjectionCapacity& storage);

        /// @brief Build a construction shard as a slice of `backing`'s columns.
        ///
        /// @details
        /// Every column views `sliceCapacity` elements of the matching `backing`
        /// column from `sliceStart` on (`ProjectionColumn::attachSlice`), so
        /// `appendLogicalBlock` writes straight into the final host projection
        /// (D-425). Construction scratch is reserved as
        /// for an owning arena. `backing` must be a backing arena that has not
        /// adopted yet and must outlive the shard; the shards of one backing
        /// arena take disjoint slices and are adopted in slice order.
        ///
        /// @param sliceCapacity Ceilings of this shard; every field positive.
        /// @param backing The backing arena whose storage this shard views.
        /// @param sliceStart First element of this shard's slice in each column.
        /// @return An empty slice arena.
        Phase2ProjectionArena(Phase2ProjectionCapacity sliceCapacity,
            Phase2ProjectionArena& backing, const Phase2ProjectionCapacity& sliceStart);

        /// @brief Heap bytes the arena's columns own (fixed at construction) —
        ///        the per-chunk host projection size.
        ///
        /// @details
        /// Sums `ProjectionColumn::ownedBytes` over every column, host-only
        /// ordinals included. A slice shard owns none: its rows live in the
        /// backing arena. The arena reserves once and never grows, so the figure
        /// is the heap this projection pins from construction to destruction.
        /// Observability only (Rule 16).
        ///
        /// @return Owned heap bytes.
        uint64_t hostBytes() const {
            const auto owned = [](const auto&... columns) {
                return (static_cast<uint64_t>(0) + ... + columns.ownedBytes());
            };
            return owned(logicalBlocks, statements, sourceStatementIndices,
                nameRecords, nameBytes, nameSlots, ruleStringRecords,
                ruleStringBytes, byteMapViews, byteMapEntries, byteMapSlots,
                byteKeyBytes, blobRecords, blobBytes, reverseMapViews,
                reverseMapEntries, reverseMapSlots, reverseKeyBytes, reverseOwners,
                podMapViews, podMapEntries, podMapSlots, podRunValues,
                mandatoryStatementKeys, metadataBytes, blobRuleOwners);
        }

        /// @brief Reset all used prefixes without releasing reserved storage.
        ///
        /// @details
        /// Clears logical blocks, statements, names, bytes, and ranking scratch.
        /// It does not shrink or replace any vector, so the next deterministic task
        /// batch reuses the exact allocation set.
        ///
        /// @return Nothing.
        /// @invariant Every public used length is zero and each physically
        ///            retained column keeps its constructor capacity.
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
        /// write. Statement rows whose signed core appears in no rule premise
        /// are excluded before upload; retained rows keep their relative order.
        ///
        /// @param body Resident logical block whose read-only Phase 2 state is read.
        /// @param analyzer Analyzer-wide immutable parameters, anchor name, and
        ///                 compiled-core categories read by evaluation.
        /// @param selectedHashMemories Bit `1 << DeviceHashMemoryKind` for every
        ///                 request memory consumed by this logical block's tasks.
        /// @param buildPrefixFilter Build the exact-prefix next-predicate maps;
        ///                 false retains the explicit projection test oracle.
        /// @return Pointer-free descriptor also appended to `logicalBlocks`.
        /// @invariant Retained rows decode byte-identically to `body` and
        ///            preserve relative statement order and local identifier values.
        DeviceLogicalBlockProjection appendLogicalBlock(
            const Memory& body,
            const ExpressionAnalyzer& analyzer,
            uint32_t selectedHashMemories = 0x0fu,
            bool buildPrefixFilter = false);

        /// @brief Translate a retained host statement ordinal to its device ordinal.
        ///
        /// @details
        /// Each block's source ordinals remain strictly ascending after stable
        /// filtering and shard concatenation. Binary search resolves a processor
        /// stump reference without uploading the source-to-device mapping.
        ///
        /// @param blockIndex Logical-block descriptor index in this arena.
        /// @param sourceIndex Original overall-list ordinal of a retained premise.
        /// @return The block-local ordinal in the filtered device statement slice.
        /// @invariant The source statement must survive the signed-core prefilter.
        NameId projectedStatementIndex(uint32_t blockIndex, NameId sourceIndex) const;

        /// @brief Most shards one `adoptShards` call adopts: the construction
        ///        shard count of the production projection.
        static constexpr uint32_t kMaxAppendedShards =
            static_cast<uint32_t>(kPhase2ProjectionConstructionShardCount);

    public:
        /// @brief Number of projected columns a shard adoption moves.
        static constexpr uint32_t kShardColumnCount = 25;

        /// @brief Destination column sizes before one shard is appended — the
        ///        bases that shard's rebased references add.
        struct ShardBases {
            uint32_t block, statement, nameRecord, nameByte, nameSlot;
            uint32_t ruleRecord, ruleByte, byteView, byteEntry, byteSlot, byteKey;
            uint32_t blobRecord, blobByte, reverseView, reverseEntry, reverseSlot;
            uint32_t reverseKey, reverseOwner, podView, podEntry, podSlot, podRun;
            uint32_t mandatoryKey, metadata;
        };

        /// @brief Take in the rows this backing arena's slice shards built, in
        ///        place, with every reference at its complete-image offset.
        ///
        /// @details
        /// The merge of a build-once projection (D-425).
        /// Computes every shard's bases up front (`currentShardBases`, then
        /// `followingShardBases` shard by shard), asserts the totals fit
        /// `capacity`, and lets `workers` take whole columns: a column moves
        /// each shard's used prefix to the end of this arena's column
        /// (`ProjectionColumn::appendMoved`, a `memmove` inside the same
        /// storage; the first shard is already in place) and rebases the moved
        /// references, shard by shard in order (`adoptShardColumn`), so no two
        /// threads write the same column. Every rebased reference also adds
        /// `streamBases`, the complete-image position of this arena's first row
        /// of the referenced column: zero for a projection uploaded whole, the
        /// prior uploaded column ends for a streamed construction window, which
        /// then needs no `rebaseForStream` pass (D-447).
        /// The result is byte-identical to building every block in one arena and
        /// rebasing it by `streamBases`. The shards end with every column empty
        /// and keep their timing, which this arena accumulates; the merge time is
        /// counted once. A backing arena adopts once.
        ///
        /// @param shards This arena's slice shards in slice order.
        /// @param count Number of shards; at most `kMaxAppendedShards`.
        /// @param firstBlocks Output: first logical-block index of each shard.
        /// @param streamBases Complete-image offset of this arena's first row in
        ///        every column; each sum with a placed base is checked to fit
        ///        `uint32_t`.
        /// @param workers Persistent threads that run the column tasks.
        /// @return Nothing.
        /// @invariant Every rebased reference addresses the byte-identical moved
        ///            value at its complete-image position, and no column
        ///            reallocates.
        void adoptShards(Phase2ProjectionArena* const* shards, uint32_t count,
            uint32_t* firstBlocks, const ShardBases& streamBases,
            Phase2ProjectionWorkers& workers);

        /// @brief Return the accumulated column ends after this window.
        ///
        /// @details Used lengths are read before rebasing; checked additions
        /// preserve the same range limits as ordinary shard adoption.
        /// @param bases Prior uploaded column ends.
        /// @return Column ends including all used window elements.
        ShardBases streamedEnd(const ShardBases& bases) const;

        /// @brief Rebase a complete window into assigned device ranges.
        ///
        /// @details Uses the same reference and slot adjustment as ordinary
        /// shard adoption; payload, local identities and order remain.
        /// @param bases Destination bases of every complete device column.
        /// @return Nothing.
        void rebaseForStream(const ShardBases& bases);

        /// @brief Retain host data from one completed construction-window upload.
        ///
        /// @details Copies block descriptors, source ordinals and name/rule
        /// lookups only. Used retained lengths must equal the assigned bases.
        /// @param source Rebased complete construction window.
        /// @param bases Destination ranges used for the uploaded window.
        /// @return Nothing.
        void appendSealingProjection(const Phase2ProjectionArena& source,
            const ShardBases& bases);

    private:
        /// @brief The destination column sizes before a shard is appended — the
        ///        bases its rebased references add.
        ///
        /// @details
        /// Reads the current size of every destination column, each asserted to fit
        /// `uint32_t` (every projected offset is 32-bit).
        ///
        /// @return The bases of the next shard appended to this arena.
        ShardBases currentShardBases() const;

        /// @brief The bases of the shard that follows `source` when `source` is
        ///        appended at `bases`.
        ///
        /// @details
        /// Adds each of `source`'s column sizes to the matching base, asserting that
        /// no sum leaves `uint32_t`. Chaining this from `currentShardBases` gives
        /// every shard's bases before any shard is copied.
        ///
        /// @param bases Bases at which `source` is appended.
        /// @param source The shard appended at `bases`.
        /// @return The bases of the next shard.
        static ShardBases followingShardBases(const ShardBases& bases,
            const Phase2ProjectionArena& source);

        /// @brief Assert that `source` adopted at `bases` fits every declared
        ///        ceiling (`capacity`), the size the device buffer is built for.
        ///
        /// @param bases Bases at which `source` is adopted.
        /// @param source The shard to adopt.
        /// @return Nothing.
        void assertShardFits(const ShardBases& bases,
            const Phase2ProjectionArena& source) const;

        /// @brief Move one column of `shard` to the end of this arena's column and
        ///        rebase the moved references by `referenceBases`.
        ///
        /// @details
        /// The per-column core of `adoptShards`: column `column`
        /// (0 .. `kShardColumnCount` - 1) is the only destination column written,
        /// and every rebase reads only the two bases, so different columns may be
        /// adopted by different threads. The column must end at its base in
        /// `bases` when this is called (shards are adopted in order).
        ///
        /// @param column Column ordinal.
        /// @param shard The slice shard whose column is moved; the column ends empty.
        /// @param bases Positions in this arena at which `shard` is placed.
        /// @param referenceBases Offsets the moved references add: `bases` plus
        ///        this arena's complete-image stream bases.
        /// @return Nothing.
        /// @invariant Every rebased reference addresses the byte-identical moved
        ///            value.
        void adoptShardColumn(uint32_t column, Phase2ProjectionArena& shard,
            const ShardBases& bases, const ShardBases& referenceBases);

        /// @brief Rebase references in an already placed column range.
        ///
        /// @details Shared by ordinary adoption and streamed uploads. Values
        /// before begin remain untouched and every offset addition is checked.
        /// @param column Host column ordinal, including source ordinals.
        /// @param begin First element of the newly placed range.
        /// @param bases Global bases of referenced columns.
        /// @return Nothing.
        void rebaseColumn(uint32_t column, uint32_t begin, const ShardBases& bases);

        /// @brief How the arena holds its column storage.
        ///
        /// @details
        /// `Owning`: the columns own their storage (tests, reference builds).
        /// `Backing`: the columns own the storage of a build-once projection and
        /// receive their rows through `adoptShards`. `Slice`: the columns view part
        /// of a backing arena's storage. Sealing retains only host lookup data.
        enum class Role : uint8_t { Owning, Backing, Slice, Sealing };

        /// @brief Assert that every field of `capacity` is positive.
        ///
        /// @param capacity Ceilings or storage sizes to check.
        /// @return Nothing.
        static void assertPositiveCapacity(const Phase2ProjectionCapacity& capacity);

        /// @brief Reserve every column to `storage` (owning and backing arenas).
        ///
        /// @param storage Elements each column reserves.
        /// @return Nothing.
        void reserveColumns(const Phase2ProjectionCapacity& storage);

        /// @brief Reserve the construction scratch for this arena's `capacity`
        ///        (owning and slice arenas).
        ///
        /// @return Nothing.
        void reserveScratch();

        /// @brief Add a shard's construction timing to this arena's totals.
        ///
        /// @param source The appended shard.
        /// @return Nothing.
        void addShardTiming(const Phase2ProjectionArena& source);

        /// @brief Append the four immutable normalized-prefix transition maps.
        ///
        /// @details
        /// Reads the already packed subkey and whole-key tables and groups their
        /// immediate prefixes into rank-ordered signed predicate unions. Singleton
        /// keys provide the immutable predicate arities. Temporary indexes use a
        /// private static-pool arena; final keys, slots and values use the existing
        /// bounded projection columns. No proof interner or source map is changed.
        ///
        /// @param block Descriptor whose first ten byte-map views are complete.
        /// @param sharedKeys This block's key-to-offset construction dictionary.
        /// @return Nothing; four derived views are appended in memory-kind order.
        /// @invariant Every existing key licenses its last predicate after its
        ///            exact ordered normalized prefix, including empty short keys.
        /// @see measurePhase2ProjectionUsage, appendLogicalBlock.
        void appendNextPredicateMaps(const DeviceLogicalBlockProjection& block,
            Phase2ProjectionKeyStore& sharedKeys);

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
        /// A later logical block may reference these packed name columns only
        /// after its names compare byte for byte, with equal parent ids, against
        /// the projected records and bytes. The comparison reads this arena's
        /// copy, never the source block, which may be deloaded once released.
        struct SharedNameProjection {
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
        /// counts and byte-identical values at every id, compared against this
        /// arena's projected records and bytes, before a descriptor may reuse
        /// the representative's record and byte slices.
        struct SharedRuleStringProjection {
            uint32_t recordOffset{ 0 };
            uint32_t recordCount{ 0 };
            uint32_t byteOffset{ 0 };
            uint32_t byteCount{ 0 };
        };

        std::vector<NameId> decodedRankScratch_;
        std::vector<uint8_t> usedAsPremiseScratch_;
        std::vector<NameId> decodedRankWorkScratch_;
        std::vector<DecodedRankRange> decodedRankRangeScratch_;
        std::vector<uint32_t> reverseRunScratch_;
        std::vector<SharedNameProjection> sharedNameProjections_;
        std::vector<SharedRuleStringProjection> sharedRuleStringProjections_;
        Role role_ = Role::Owning;
        bool adopted_ = false;
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

    /// @brief One executor part's task usage from its logical block's
    ///        stump-free usage and the part's stump count.
    ///
    /// @details
    /// `measurePhase2TaskProjectionUsage` with a stump count of zero depends only
    /// on the logical block, so the CUDA route records it once per burst at the
    /// end of the block's phase 1 and derives every part's usage here: a
    /// scheduled task (`tasks != 0`) carries the part's stump count, an
    /// unscheduled one stays all zero, and counter-example mode owns no stumps.
    ///
    /// @param stumpFree The block's usage measured with a stump count of zero.
    /// @param stumpCount Number of stumps owned by this executor part.
    /// @param counterExampleMode Whether this is the unsplit counter-example route.
    /// @return Exactly `measurePhase2TaskProjectionUsage(body, stumpCount,
    ///         counterExampleMode)` for the block `stumpFree` was measured on.
    /// @invariant Asserts `stumpFree.stumps == 0`, and `stumpCount == 0` in
    ///            counter-example mode.
    Phase2TaskProjectionUsage phase2TaskUsageWithStumps(
        const Phase2TaskProjectionUsage& stumpFree,
        uint32_t stumpCount,
        bool counterExampleMode);

    /// @brief The hash memories one logical block's executable request batches
    ///        read — the selection mask of `measurePhase2ProjectionUsage`.
    ///
    /// @details
    /// Bit `1 << DeviceHashMemoryKind` for every memory named by a batch that
    /// `measurePhase2TaskProjectionUsage` counts: working rules (working map and
    /// local statements), overall (a delta, or externals with local
    /// statements), local (local map with externals) and the local delta map.
    /// Counter-example mode selects the overall memory only. Reads only the
    /// emptiness of the block's containers.
    ///
    /// @param body Resident logical block.
    /// @param counterExampleMode Whether this is the unsplit counter-example route.
    /// @return The selection mask; zero when the block has no executable batch.
    uint32_t selectPhase2HashMemories(const Memory& body, bool counterExampleMode);

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

        /// @brief Heap bytes the four task columns reserve (fixed at
        ///        construction). Observability only (Rule 16).
        ///
        /// @return Reserved heap bytes.
        uint64_t hostBytes() const {
            return vectorCapacityBytes(tasks, batches, terms, stumps);
        }

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
    /// ceiling, configured eligible-row limit, and the processor filter's whole-key
    /// widening flag. The resident logical-block descriptor supplies the statement slice and name ranks.
    struct DevicePhase2FilterCall {
        uint32_t logicalBlockIndex{ 0 };
        DeviceHashMemoryKind memory{ DeviceHashMemoryKind::overall };
        int32_t maximumIterationNumberVariable{ 0 };
        uint32_t alsoAcceptFullKeys{ 0 };
        uint32_t statementCount{ 0 };
        uint32_t maximumFilteredStatements{ 0 };
    };

    /// @brief Fixed ceilings for one global statement filter/sort sweep.
    struct Phase2FilterScheduleCapacity {
        uint32_t calls{ 0 };
        uint64_t examinedRows{ 0 };
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
        uint64_t uniqueExaminedRows{ 0 };
        uint64_t duplicateExaminedRows{ 0 };
        uint32_t maximumClassMultiplicity{ 0 };
    };

    // Each class owns a separate sort span; both key fields retain full width.
    inline constexpr uint32_t kDeviceFilterStatementIndexBits = 32;
    inline constexpr uint32_t kDeviceFilterStatementIndexMask = UINT32_MAX;

    static_assert(std::is_trivially_copyable_v<DevicePhase2FilterCall>);
    static_assert(sizeof(DevicePhase2FilterCall) == 24);

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
        uint64_t examinedRows{ 0 };
        uint64_t classExaminedRows{ 0 };

        /// @brief Heap bytes the filter-schedule columns reserve (fixed at
        ///        construction). Observability only (Rule 16).
        ///
        /// @return Reserved heap bytes.
        uint64_t hostBytes() const {
            return vectorCapacityBytes(calls, classes, callClassIndices,
                classMultiplicities, classSlots);
        }

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
        /// Copies only scalar semantic selectors, interns the complete six-field
        /// signature through the fixed open-address table, and appends the class
        /// index beside the original processor-order call. Statement counts are
        /// accumulated for both original and unique-class work.
        ///
        /// @param logicalBlockIndex Resident logical-block projection index.
        /// @param memory Hash-memory registry supplying whole and subkey maps.
        /// @param maximumIterationNumberVariable Inclusive iteration ceiling.
        /// @param alsoAcceptFullKeys Bivalent processor whole-key widening flag.
        /// @param statementCount Full resident statement slice examined by the call.
        /// @param maximumFilteredStatements Positive eligible-row limit fitting the position codes.
        /// @return Pointer-free descriptor also appended to `calls`.
        /// @invariant Call order exactly matches the processor schedule; class
        ///            order is first occurrence; neither row total exceeds its
        ///            fixed ceiling.
        DevicePhase2FilterCall appendCall(
            uint32_t logicalBlockIndex,
            DeviceHashMemoryKind memory,
            int32_t maximumIterationNumberVariable,
            uint32_t alsoAcceptFullKeys,
            uint32_t statementCount,
            uint32_t maximumFilteredStatements = ProverParameters{}.maxNumberFilteredStatements);

        /// @brief Report exact filter-signature reuse in the production schedule.
        ///
        /// @details
        /// Reads the exact classes and multiplicities already built from logical
        /// block, hash-memory kind, iteration ceiling, whole-key widening flag,
        /// statement count, and retained-row limit. The compact scan allocates no
        /// storage and never changes call order, row accounting, or a proof input.
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

        /// @brief Heap bytes the growth-call schedule reserves (fixed at
        ///        construction). Observability only (Rule 16).
        ///
        /// @return Reserved heap bytes.
        uint64_t hostBytes() const { return vectorCapacityBytes(calls); }

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
    /// `frontierStoreRecords` (at least `frontierRecords`) sizes the two node
    /// buffers of the pass-local frontier store: the windowed driver keeps a
    /// whole level and its children there and spills only the rows of a level
    /// that outgrows a buffer to host pages.
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
        uint32_t frontierStoreRecords{ 0 };
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
        67108864,
        4194304 };

    /// @brief Common full-run growth limits retain measured corpus headroom without unused oversized pools.
    inline constexpr Phase2GrowthCapacity kFullRunPhase2GrowthCapacity{
        4096,
        1048576,
        3670016,
        6291456,
        524288,
        109051904,
        29360128,
        6291456,
        16777216,
        3670016 };

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

    /// @brief Contiguous slice of a stable growth-node index list.
    ///
    /// @details The slice covers every positive candidate span overlapping a
    /// requested window. Empty spans between its boundaries may remain.
    struct DeviceGrowthNodeWindow {
        uint32_t begin{0};
        uint32_t count{0};
    };

    /// @brief Find the node-index slice covering one candidate-attempt window.
    ///
    /// @details Two binary searches use the monotonic cumulative candidate
    /// boundaries and the ascending indices produced by stable kind selection.
    /// Spans touching only a window boundary are excluded. An empty kind list
    /// has a defined empty slice and does not require backing arrays.
    ///
    /// @param indices Ascending original node indices, or null for an empty list.
    /// @param nodeCount Number of entries in the selected kind list.
    /// @param starts Cumulative candidate starts indexed by original node.
    /// @param ends Cumulative candidate ends indexed by original node.
    /// @param windowBegin Inclusive candidate ordinal of a nonempty window.
    /// @param windowEnd Exclusive candidate ordinal of that window.
    /// @return Consecutive kind-list entries covering all overlapping candidates.
    /// @invariant Starts and ends are nondecreasing in original node order.
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline DeviceGrowthNodeWindow growthCandidateNodeWindow(
        const uint32_t* indices, uint32_t nodeCount, const uint64_t* starts,
        const uint64_t* ends, uint64_t windowBegin, uint64_t windowEnd) {
        assert(windowBegin < windowEnd);
        assert(nodeCount == 0 || (indices != nullptr && starts != nullptr && ends != nullptr));
        uint32_t low = 0, high = nodeCount;
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2;
            if (ends[indices[middle]] <= windowBegin) low = middle + 1;
            else high = middle;
        }
        const uint32_t begin = low;
        high = nodeCount;
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2;
            if (starts[indices[middle]] < windowEnd) low = middle + 1;
            else high = middle;
        }
        return {begin, low - begin};
    }

    static_assert(sizeof(DeviceGrowthNodeWindow) == 8);
    static_assert(std::is_trivially_copyable_v<DeviceGrowthNodeWindow>);

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
        int32_t nextPredicateEntry{ -2 };
        uint32_t candidateCount{ 0 };
    };

    /// @brief Select the first iterative occurrence of each full argument identity.
    ///
    /// @details
    /// Builds the statement-only part of the secondary-variable gate. Negative
    /// iterations contribute nothing; repeated iterative identities contribute
    /// once, at their first iterative position. The caller removes recursion
    /// products using the immutable logical-block map before caching this mask.
    ///
    /// @param expression Immutable statement whose active arguments are inspected.
    /// @return One bit per distinct iterative argument position.
    /// @invariant No bit at or above the statement arity is set.
    /// @see IntEncodedExpr, DevicePhase2GrowthParameters.
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline uint32_t growthDistinctSecondaryArgumentMask(
        const IntEncodedExpr& expression) {
        static_assert(ExecutionParameters::MAX_ARITY <= 16);
        assert(expression.arity >= 0);
        assert(expression.arity <= ExecutionParameters::MAX_ARITY);
        uint32_t mask = 0;
        for (int32_t argument = 0; argument < expression.arity; ++argument) {
            if (expression.argIteration[argument] <= -1) continue;
            bool repeated = false;
            for (int32_t prior = 0; prior < argument; ++prior) {
                if (expression.argIteration[prior] > -1
                    && expression.argFullId[prior] == expression.argFullId[argument]) {
                    repeated = true;
                    break;
                }
            }
            if (!repeated) mask |= uint32_t{1} << argument;
        }
        return mask;
    }

    /// @brief Argument requirements a continuation word places on one appended premise.
    ///
    /// @details
    /// `polarityMask` keeps the licensed signs (bit 0 positive, bit 1 negative).
    /// Each used gate names an argument slot of the appended premise that every
    /// key of the transition binds to the same normalized variable of the prefix;
    /// `slot < 0` marks an unused gate. Variables count from one in the prefix's
    /// first-appearance order, like the normalized key payload.
    ///
    /// @invariant A used gate has `0 <= slot < MAX_ARITY` and `variable >= 1`.
    struct NextPredicateArgumentGate {
        uint32_t polarityMask{ 0 };
        int32_t slot[2]{ -1, -1 };
        NameId variable[2]{ 0, 0 };
    };

    /// @brief Pack a polarity mask and up to two bound argument slots into one
    ///        continuation word of the exact-prefix transition index.
    ///
    /// @details
    /// Layout: bits 0-1 polarity mask; bits 2-6 first slot plus one (zero =
    /// unused); bits 7-15 its variable; bits 16-20 second slot plus one; bits
    /// 21-29 its variable. The word replaces the former bare polarity mask in the
    /// eight-byte `(predicate, word)` transition entries, so the projection
    /// image keeps its size.
    ///
    /// @param polarityMask Licensed signs, one or both of bits 0 and 1.
    /// @param firstSlot First bound slot, or -1.
    /// @param firstVariable Its prefix variable (ignored when unused).
    /// @param secondSlot Second bound slot, or -1 (requires a used first slot).
    /// @param secondVariable Its prefix variable (ignored when unused).
    /// @return The packed continuation word.
    /// @invariant Slots below `MAX_ARITY`, variables in 1..`MAX_KEY_SLOTS` (asserted).
    /// @see unpackNextPredicateWord, Phase2ProjectionArena::appendNextPredicateMaps.
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline uint32_t packNextPredicateWord(uint32_t polarityMask,
        int32_t firstSlot, NameId firstVariable, int32_t secondSlot, NameId secondVariable) {
        assert(polarityMask >= 1 && polarityMask <= 3);
        assert(firstSlot >= -1 && firstSlot < ExecutionParameters::MAX_ARITY);
        assert(secondSlot >= -1 && secondSlot < ExecutionParameters::MAX_ARITY);
        assert(secondSlot < 0 || firstSlot >= 0);
        uint32_t word = polarityMask;
        if (firstSlot >= 0) {
            assert(firstVariable >= 1 && firstVariable <= ExecutionParameters::MAX_KEY_SLOTS);
            word |= static_cast<uint32_t>(firstSlot + 1) << 2;
            word |= static_cast<uint32_t>(firstVariable) << 7;
        }
        if (secondSlot >= 0) {
            assert(secondVariable >= 1 && secondVariable <= ExecutionParameters::MAX_KEY_SLOTS);
            word |= static_cast<uint32_t>(secondSlot + 1) << 16;
            word |= static_cast<uint32_t>(secondVariable) << 21;
        }
        return word;
    }

    /// @brief Decode a continuation word of the exact-prefix transition index.
    ///
    /// @details Inverse of `packNextPredicateWord`.
    ///
    /// @param word Packed continuation word.
    /// @return Its polarity mask and argument gates.
    /// @invariant The polarity mask is nonzero (asserted).
    /// @see packNextPredicateWord.
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline NextPredicateArgumentGate unpackNextPredicateWord(uint32_t word) {
        NextPredicateArgumentGate gate{};
        gate.polarityMask = word & 3u;
        assert(gate.polarityMask != 0);
        const uint32_t first = (word >> 2) & 31u;
        const uint32_t second = (word >> 16) & 31u;
        if (first != 0) {
            gate.slot[0] = static_cast<int32_t>(first) - 1;
            gate.variable[0] = static_cast<NameId>((word >> 7) & 511u);
            assert(gate.variable[0] >= 1);
        }
        if (second != 0) {
            gate.slot[1] = static_cast<int32_t>(second) - 1;
            gate.variable[1] = static_cast<NameId>((word >> 21) & 511u);
            assert(gate.variable[1] >= 1);
        }
        return gate;
    }

    /// @brief Fold one key's appended-premise arguments into a transition's
    ///        per-slot prefix bindings.
    ///
    /// @details
    /// A slot stays bound to variable `v` only while every key of the transition
    /// gives that slot the normalized identifier `v` of a prefix variable
    /// (`v <= prefixVariables`); a fresh variable or a disagreeing key marks the
    /// slot -1 for good. The first key initializes the slots. A candidate whose
    /// argument at a still-bound slot differs from that prefix variable's value
    /// normalizes differently from every key of the transition, so it can match
    /// none of them.
    ///
    /// @param bound Per-slot bindings, `MAX_ARITY` entries, updated in place.
    /// @param arity Arity of the appended predicate.
    /// @param arguments Normalized argument identifiers of the appended premise.
    /// @param prefixVariables Number of distinct variables in the prefix.
    /// @param first Whether this is the transition's first key.
    /// @return Nothing.
    /// @invariant Slots at or above @p arity are -1 after the first key.
    /// @see packNextPredicateWord, Phase2ProjectionArena::appendNextPredicateMaps.
    inline void mergeNextPremiseBindings(int16_t* bound, NameId arity,
        const NameId* arguments, NameId prefixVariables, bool first) {
        assert(arity >= 0 && arity <= ExecutionParameters::MAX_ARITY);
        for (NameId slot = 0; slot < ExecutionParameters::MAX_ARITY; ++slot) {
            if (slot >= arity) {
                assert(first || bound[slot] == -1);
                bound[slot] = -1;
                continue;
            }
            const NameId argument = arguments[slot];
            assert(argument >= 1);
            const bool prefixBound = argument <= prefixVariables;
            if (first) bound[slot] = prefixBound ? static_cast<int16_t>(argument) : int16_t{ -1 };
            else if (bound[slot] != argument || !prefixBound) bound[slot] = -1;
        }
    }

    /// @brief Argument values a transition's continuation word requires of every
    ///        appended statement of one growth node.
    ///
    /// @details
    /// `admits` applies the licensed polarity and, for each bound slot of the
    /// word, compares the statement's argument identifier with the prefix
    /// variable's value from the node's distinct normalization-variable pool. A
    /// statement failing either test normalizes to a key outside the transition,
    /// so `preparePooledGrowthCandidate` would reject it at the key probe; the
    /// test only rejects it before the per-candidate gates. No candidate
    /// position, ordinal or accepted event changes.
    struct GrowthArgumentGate {
        uint32_t polarityMask;
        int32_t slot[2];
        NameId required[2];

        /// @brief Whether a statement can extend the node under this transition.
        /// @param expression Candidate statement.
        /// @return False when the sign or a bound argument rules it out.
#ifdef __CUDACC__
        __host__ __device__
#endif
        bool admits(const IntEncodedExpr& expression) const {
            if ((polarityMask & (1u << expression.negation)) == 0) return false;
            for (int32_t gate = 0; gate < 2; ++gate) {
                if (slot[gate] < 0) continue;
                assert(slot[gate] < expression.arity);
                if (expression.argId[slot[gate]] != required[gate]) return false;
            }
            return true;
        }
    };

    /// @brief Resolve a continuation word against one node's prefix variables.
    ///
    /// @details Decodes the word (`unpackNextPredicateWord`) and reads each bound
    /// variable's argument identifier from the pooled distinct
    /// normalization-variable run of the node's prefix (first-appearance order,
    /// the normalized key's numbering).
    ///
    /// @param word Transition continuation word.
    /// @param prefix The node's immutable pooled prefix header.
    /// @param prefixVariables Fixed distinct normalization-variable pool.
    /// @return The node's argument gate for this predicate run.
    /// @invariant Every bound variable lies inside the prefix's variable run (asserted).
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline GrowthArgumentGate growthArgumentGate(uint32_t word,
        const DeviceGrowthPrefix& prefix, const NameId* prefixVariables) {
        const NextPredicateArgumentGate decoded = unpackNextPredicateWord(word);
        GrowthArgumentGate gate{ decoded.polarityMask, { -1, -1 }, { 0, 0 } };
        for (int32_t index = 0; index < 2; ++index) {
            if (decoded.slot[index] < 0) continue;
            assert(decoded.variable[index] >= 1
                && decoded.variable[index] <= static_cast<NameId>(prefix.variableCount));
            gate.slot[index] = decoded.slot[index];
            gate.required[index] = prefixVariables[prefix.variableOffset
                + static_cast<uint32_t>(decoded.variable[index] - 1)];
        }
        return gate;
    }

    /// @brief Locate one predicate's retained statement positions after a cursor.
    ///
    /// @details
    /// Binary searches the existing decoded-name-rank sort without changing its
    /// first-row retention or stable statement positions. A missing predicate or
    /// a cursor past its final row produces a defined empty range.
    ///
    /// @param sortedKeys Original packed rank and statement-index keys.
    /// @param offset Beginning of the immutable filter-class span.
    /// @param count Number of retained rows in that span.
    /// @param startPosition Inclusive original-position cursor.
    /// @param predicateRank Exact decoded-name rank to select.
    /// @return Original-position range belonging to that predicate.
    /// @invariant Every selected position is at least startPosition and below count.
    /// @see DeviceGrowthPrefix, growthCandidateNodeWindow.
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline DeviceGrowthNodeWindow growthPredicateRange(
        const uint64_t* sortedKeys, uint32_t offset, uint32_t count,
        uint32_t startPosition, uint32_t predicateRank) {
        assert(startPosition <= count);
        assert(count == 0 || sortedKeys != nullptr);
        uint32_t low = startPosition, high = count;
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2;
            const uint32_t rank = static_cast<uint32_t>(
                sortedKeys[offset + middle] >> kDeviceFilterStatementIndexBits);
            if (rank < predicateRank) low = middle + 1;
            else high = middle;
        }
        const uint32_t begin = low;
        high = count;
        while (low < high) {
            const uint32_t middle = low + (high - low) / 2;
            const uint32_t rank = static_cast<uint32_t>(
                sortedKeys[offset + middle] >> kDeviceFilterStatementIndexBits);
            if (rank <= predicateRank) low = middle + 1;
            else high = middle;
        }
        return DeviceGrowthNodeWindow{begin, low - begin};
    }

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

    /// @brief Read one exact processor-walk path ordering component.
    ///
    /// @details Leaf positions ascend, while carried prefix positions descend
    /// above the leaf range. Zero codes denote absent stump elements. Host
    /// continuation metadata and CUDA ordering share this representation.
    ///
    /// @param event Accepted event with its original filtered-position codes.
    /// @param component Path component below MAX_EXPRESSIONS.
    /// @return Unsigned radix component used by the original processor walk.
    /// @invariant Every present position has a nonzero uint16_t code; zero is reserved.
#ifdef __CUDACC__
    __host__ __device__
#endif
    inline uint32_t growthEventPathComponent(const DeviceAcceptedGrowthEvent& event,
        uint32_t component) {
        assert(component < ExecutionParameters::MAX_EXPRESSIONS);
        if (component >= static_cast<uint32_t>(event.count)) return 0;
        const uint16_t code = event.filteredPositionCodes[component];
        const uint32_t position = code == 0 ? 0u : static_cast<uint32_t>(code - 1);
        assert(position < ExecutionParameters::MAX_FILTERED_POSITION_CODE);
        return component + 1 == static_cast<uint32_t>(event.count)
            ? position : ExecutionParameters::MAX_FILTERED_POSITION_CODE + 1u
                + (ExecutionParameters::MAX_FILTERED_POSITION_CODE - 1u - position);
    }

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
    static_assert(sizeof(Phase2GrowthCapacity) == 40);
    static_assert(sizeof(DeviceGrowthNode) == 56);
    static_assert(sizeof(DeviceGrowthPrefix) == 40);
    static_assert(sizeof(DeviceGrowthCandidateAttempt) == 16);
    static_assert(sizeof(DeviceAcceptedGrowthEvent) == 64);
    static_assert(sizeof(DeviceRawGrowthRequest) == 72);

}  // namespace gl::gpu
