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

#include "phase2_projection.hpp"

#include "../memory.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace gl::gpu {

    /// @brief Return the audited fixed projection ceilings for one CUDA route.
    ///
    /// @details
    /// Constructs the complete FTA-shortcut or full-run projection tuple without
    /// allocating storage. Both profiles use the same physical task-bearing
    /// logical-block ceiling; only their measured column ceilings differ.
    ///
    /// @param profile Fixed corpus profile selected for this process.
    /// @return Complete immutable capacity tuple for a projection arena.
    /// @invariant `logicalBlocks` equals `kMaxProjectedBlocksPerChunk`.
    Phase2ProjectionCapacity phase2ProjectionCapacityFor(
        Phase2ProjectionProfile profile) {
        Phase2ProjectionCapacity capacity{};
        switch (profile) {
        case Phase2ProjectionProfile::ftaShortcut:
            capacity = kFtaShortcutPhase2ProjectionCapacity;
            break;
        case Phase2ProjectionProfile::fullRun:
            capacity = kFullRunPhase2ProjectionCapacity;
            break;
        }
        assert(capacity.logicalBlocks == kMaxProjectedBlocksPerChunk);
        assert(capacity.statements > 0);
        return capacity;
    }

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
        uint32_t selectedHashMemories) {
        assert(selectedHashMemories != 0);
        assert((selectedHashMemories & ~0x0fu) == 0);
        Phase2ProjectionUsage usage{};
        usage.logicalBlocks = 1;
        usage.statements = static_cast<uint64_t>(
            body.intEncodedStatementsCount());
        const int32_t nameCount = body.nameMap.nameCount();
        assert(nameCount >= 0);
        usage.nameRecords = static_cast<uint64_t>(nameCount) + 1;
        const int64_t nameBytes = body.nameMap.names->contentBytesFrom(0);
        assert(nameBytes >= 0);
        usage.nameBytes = static_cast<uint64_t>(nameBytes);

        const int32_t ruleStringCount = body.ruleInterner.internedCount();
        assert(ruleStringCount >= 0);
        usage.ruleStringRecords = static_cast<uint64_t>(ruleStringCount) + 1;
        assert(body.ruleInterner.table != nullptr);
        const int64_t ruleStringBytes =
            body.ruleInterner.table->keyStore().logicalByteCount();
        assert(ruleStringBytes >= 0);
        usage.ruleStringBytes = static_cast<uint64_t>(ruleStringBytes);

        const auto slotsFor = [](int32_t count) -> uint64_t {
            assert(count >= 0);
            if (count == 0) return 0;
            uint64_t slots = 2;
            const uint64_t required = static_cast<uint64_t>(count) * 2;
            while (slots < required) {
                assert(slots <= (std::numeric_limits<uint64_t>::max() >> 1));
                slots <<= 1;
            }
            return slots;
        };
        const auto measureByteSet = [&](const auto& map) {
            ++usage.byteMapViews;
            const int32_t count = map.count();
            assert(count >= 0);
            usage.byteMapEntries += static_cast<uint64_t>(count);
            usage.byteMapSlots += slotsFor(count);
            const int64_t keyBytes =
                map.inner().keyStore().logicalByteCount();
            assert(keyBytes >= 0);
            usage.byteKeyBytes += static_cast<uint64_t>(keyBytes);
        };
        const auto measureByteBlobMap = [&](const auto& map) {
            ++usage.byteMapViews;
            const int32_t count = map.count();
            assert(count >= 0);
            usage.byteMapEntries += static_cast<uint64_t>(count);
            usage.byteMapSlots += slotsFor(count);
            const int64_t keyBytes = map.inner().keyStore().logicalByteCount();
            assert(keyBytes >= 0);
            usage.byteKeyBytes += static_cast<uint64_t>(keyBytes);
            const int32_t blobCount = map.inner().blobCount();
            const int32_t blobBytes = map.inner().poolByteCount();
            assert(blobCount >= 0 && blobBytes >= 0);
            usage.blobRecords += static_cast<uint64_t>(blobCount);
            usage.blobBytes += static_cast<uint64_t>(blobBytes);
        };
        const HashMemory* const memories[4] = {
            &body.overallHashMemory,
            &body.localHashMemory,
            &body.localHashMemoryDelta,
            &body.workingMemory
        };
        for (uint32_t memoryIndex = 0; memoryIndex < 4; ++memoryIndex) {
            if ((selectedHashMemories & (1u << memoryIndex)) == 0) {
                usage.byteMapViews += 2;
                continue;
            }
            const HashMemory* memory = memories[memoryIndex];
            measureByteSet(memory->normalizedEncodedKeys);
            measureByteBlobMap(memory->normalizedEncodedSubkeys);
        }
        measureByteBlobMap(body.overallHashMemory.encodedMap);
        const auto& remainingArgs =
            body.overallHashMemory.remainingArgsNormalizedEncodedMap;
        measureByteBlobMap(remainingArgs);

        usage.reverseMapViews = 1;
        const int32_t reverseEdgeCount = remainingArgs.inner().blobCount();
        const int32_t reverseEdgeBytes = remainingArgs.inner().poolByteCount();
        assert(reverseEdgeCount >= 0 && reverseEdgeBytes >= 0);
        usage.reverseMapEntries += static_cast<uint64_t>(reverseEdgeCount);
        usage.reverseOwners += static_cast<uint64_t>(reverseEdgeCount);
        usage.reverseKeyBytes += static_cast<uint64_t>(reverseEdgeBytes);
        assert(usage.reverseMapEntries
            <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max()));
        usage.reverseMapSlots = slotsFor(
            static_cast<int32_t>(usage.reverseMapEntries));

        const auto measurePodSet = [&](const auto& map) {
            ++usage.podMapViews;
            const int32_t count = map.count();
            assert(count >= 0);
            usage.podMapEntries += static_cast<uint64_t>(count);
            usage.podMapSlots += slotsFor(count);
        };
        const auto measurePodRunMap = [&](const auto& map) {
            measurePodSet(map);
            const int32_t count = map.count();
            for (int32_t id = 1; id <= count; ++id) {
                const int32_t runLength = map.runLen(id);
                assert(runLength >= 0);
                usage.podRunValues += static_cast<uint64_t>(runLength);
            }
        };
        const auto measureStatementVectorSet = [&](const auto& statements) {
            ++usage.podMapViews;
            const int32_t count = statements.size();
            assert(count >= 0);
            usage.podMapEntries += static_cast<uint64_t>(count);
            usage.podMapSlots += slotsFor(count);
        };
        measurePodSet(body.overallHashMemory.productsOfRecursionIds);
        measurePodSet(body.intKnownStatements);
        measurePodSet(body.intLocalEncodedStatementsSet);
        measureStatementVectorSet(body.intLocalEncodedStatementsDelta);
        measureStatementVectorSet(body.intExternalStatements);
        measurePodRunMap(body.intStatementLevelsMap);
        measurePodSet(body.intValidityNamesToFilter);
        measurePodSet(body.frozenOrBranches);
        measurePodRunMap(body.intToBeProved);
        measurePodSet(body.canBeSentIds);
        measurePodSet(body.canBeSentMarkerIds);

        usage.mandatoryStatementKeys = static_cast<uint64_t>(
            body.intLocalEncodedStatements.size())
            + static_cast<uint64_t>(body.intLocalEncodedStatementsDelta.size())
            + static_cast<uint64_t>(body.intExternalStatements.size());
        const StrSpan expressionKey = body.exprKeyView();
        assert(expressionKey.len >= 0);
        usage.metadataBytes = static_cast<uint64_t>(expressionKey.len)
            + static_cast<uint64_t>(analyzer.anchorInfo.name.size());
        usage.nameSlots = slotsFor(nameCount);
        return usage;
    }

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
        bool counterExampleMode) {
        Phase2TaskProjectionUsage usage{};
        if (counterExampleMode) {
            assert(stumpCount == 0);
            usage.tasks = 1;
            usage.batches = 1;
            return usage;
        }

        if (!body.workingMemory.encodedMap.empty()
            && !body.intLocalEncodedStatements.empty()) {
            ++usage.batches;
            ++usage.terms;
        }
        const bool hasNewThisBurst =
            !body.intLocalEncodedStatementsDelta.empty()
            || (!body.intExternalStatements.empty()
                && !body.intLocalEncodedStatements.empty());
        if (hasNewThisBurst) {
            ++usage.batches;
            usage.terms += 2;
        }
        if (!body.localHashMemory.encodedMap.empty()
            && !body.intExternalStatements.empty()) {
            ++usage.batches;
            ++usage.terms;
        }
        if (!body.localHashMemoryDelta.encodedMap.empty())
            ++usage.batches;
        if (usage.batches != 0) {
            usage.tasks = 1;
            usage.stumps = stumpCount;
        }
        return usage;
    }

    /// @brief Reserve every projection array to an immutable capacity ceiling.
    ///
    /// @details
    /// Asserts positive logical-block and name-record capacity, reserves all
    /// semantic arrays plus decoded-rank scratch once, and leaves every used
    /// length zero. Later appends assert before crossing any ceiling.
    ///
    /// @param fixedCapacity Element and byte ceilings for this arena lifetime.
    /// @return A reusable empty arena with all storage reserved.
    /// @invariant Vector capacities are at least their corresponding declared
    ///            ceilings for the arena's entire lifetime.
    Phase2ProjectionArena::Phase2ProjectionArena(
        Phase2ProjectionCapacity fixedCapacity)
        : capacity(fixedCapacity) {
        assert(capacity.logicalBlocks > 0);
        assert(capacity.statements > 0);
        assert(capacity.nameRecords > 0);
        assert(capacity.nameBytes > 0);
        assert(capacity.nameSlots > 0);
        assert(capacity.ruleStringRecords > 0);
        assert(capacity.ruleStringBytes > 0);
        assert(capacity.byteMapViews > 0);
        assert(capacity.byteMapEntries > 0);
        assert(capacity.byteMapSlots > 0);
        assert(capacity.byteKeyBytes > 0);
        assert(capacity.blobRecords > 0);
        assert(capacity.blobBytes > 0);
        assert(capacity.reverseMapViews > 0);
        assert(capacity.reverseMapEntries > 0);
        assert(capacity.reverseMapSlots > 0);
        assert(capacity.reverseKeyBytes > 0);
        assert(capacity.reverseOwners > 0);
        assert(capacity.podMapViews > 0);
        assert(capacity.podMapEntries > 0);
        assert(capacity.podMapSlots > 0);
        assert(capacity.podRunValues > 0);
        assert(capacity.mandatoryStatementKeys > 0);
        assert(capacity.metadataBytes > 0);
        logicalBlocks.reserve(capacity.logicalBlocks);
        statements.reserve(capacity.statements);
        nameRecords.reserve(capacity.nameRecords);
        nameBytes.reserve(capacity.nameBytes);
        nameSlots.reserve(capacity.nameSlots);
        ruleStringRecords.reserve(capacity.ruleStringRecords);
        ruleStringBytes.reserve(capacity.ruleStringBytes);
        byteMapViews.reserve(capacity.byteMapViews);
        byteMapEntries.reserve(capacity.byteMapEntries);
        byteMapSlots.reserve(capacity.byteMapSlots);
        byteKeyBytes.reserve(capacity.byteKeyBytes);
        blobRecords.reserve(capacity.blobRecords);
        blobBytes.reserve(capacity.blobBytes);
        reverseMapViews.reserve(capacity.reverseMapViews);
        reverseMapEntries.reserve(capacity.reverseMapEntries);
        reverseMapSlots.reserve(capacity.reverseMapSlots);
        reverseKeyBytes.reserve(capacity.reverseKeyBytes);
        reverseOwners.reserve(capacity.reverseOwners);
        podMapViews.reserve(capacity.podMapViews);
        podMapEntries.reserve(capacity.podMapEntries);
        podMapSlots.reserve(capacity.podMapSlots);
        podRunValues.reserve(capacity.podRunValues);
        mandatoryStatementKeys.reserve(capacity.mandatoryStatementKeys);
        metadataBytes.reserve(capacity.metadataBytes);
        decodedRankScratch_.reserve(capacity.nameRecords);
        decodedRankWorkScratch_.reserve(capacity.nameRecords);
        decodedRankRangeScratch_.reserve(capacity.nameRecords);
        reverseRunScratch_.reserve(capacity.reverseMapEntries);
        sharedNameProjections_.reserve(capacity.logicalBlocks);
        sharedRuleStringProjections_.reserve(capacity.logicalBlocks);
    }

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
    void Phase2ProjectionArena::clear() {
        timing = Phase2ProjectionTiming{};
        logicalBlocks.clear();
        statements.clear();
        nameRecords.clear();
        nameBytes.clear();
        nameSlots.clear();
        ruleStringRecords.clear();
        ruleStringBytes.clear();
        byteMapViews.clear();
        byteMapEntries.clear();
        byteMapSlots.clear();
        byteKeyBytes.clear();
        blobRecords.clear();
        blobBytes.clear();
        reverseMapViews.clear();
        reverseMapEntries.clear();
        reverseMapSlots.clear();
        reverseKeyBytes.clear();
        reverseOwners.clear();
        podMapViews.clear();
        podMapEntries.clear();
        podMapSlots.clear();
        podRunValues.clear();
        mandatoryStatementKeys.clear();
        metadataBytes.clear();
        decodedRankScratch_.clear();
        decodedRankWorkScratch_.clear();
        decodedRankRangeScratch_.clear();
        reverseRunScratch_.clear();
        sharedNameProjections_.clear();
        sharedRuleStringProjections_.clear();
        assert(logicalBlocks.capacity() >= capacity.logicalBlocks);
        assert(statements.capacity() >= capacity.statements);
        assert(nameRecords.capacity() >= capacity.nameRecords);
        assert(nameBytes.capacity() >= capacity.nameBytes);
        assert(nameSlots.capacity() >= capacity.nameSlots);
        assert(ruleStringRecords.capacity() >= capacity.ruleStringRecords);
        assert(ruleStringBytes.capacity() >= capacity.ruleStringBytes);
        assert(byteMapViews.capacity() >= capacity.byteMapViews);
        assert(byteMapEntries.capacity() >= capacity.byteMapEntries);
        assert(byteMapSlots.capacity() >= capacity.byteMapSlots);
        assert(byteKeyBytes.capacity() >= capacity.byteKeyBytes);
        assert(blobRecords.capacity() >= capacity.blobRecords);
        assert(blobBytes.capacity() >= capacity.blobBytes);
        assert(reverseMapViews.capacity() >= capacity.reverseMapViews);
        assert(reverseMapEntries.capacity() >= capacity.reverseMapEntries);
        assert(reverseMapSlots.capacity() >= capacity.reverseMapSlots);
        assert(reverseKeyBytes.capacity() >= capacity.reverseKeyBytes);
        assert(reverseOwners.capacity() >= capacity.reverseOwners);
        assert(podMapViews.capacity() >= capacity.podMapViews);
        assert(podMapEntries.capacity() >= capacity.podMapEntries);
        assert(podMapSlots.capacity() >= capacity.podMapSlots);
        assert(podRunValues.capacity() >= capacity.podRunValues);
        assert(mandatoryStatementKeys.capacity()
            >= capacity.mandatoryStatementKeys);
        assert(metadataBytes.capacity() >= capacity.metadataBytes);
        assert(decodedRankScratch_.capacity() >= capacity.nameRecords);
        assert(decodedRankWorkScratch_.capacity() >= capacity.nameRecords);
        assert(decodedRankRangeScratch_.capacity() >= capacity.nameRecords);
        assert(reverseRunScratch_.capacity() >= capacity.reverseMapEntries);
        assert(sharedNameProjections_.capacity() >= capacity.logicalBlocks);
        assert(sharedRuleStringProjections_.capacity()
            >= capacity.logicalBlocks);
    }

    /// @brief Append one resident logical block as pointer-free device slices.
    ///
    /// @details
    /// Copies statements, names, rule strings, selected hash-memory
    /// whole-key/subkey tables, fixed zero-length views for unselected hash
    /// memories, the overall evaluation tables, the derived remaining-argument
    /// reverse index, plain-data maps, mandatory
    /// keys, and scalar metadata. It builds fixed-load lookup slots and
    /// decoded-name ranks using constructor-reserved scratch. Byte-identical
    /// name/id/parent tables and rule-string/id tables within one arena share
    /// their immutable packed slices after an exact row comparison; logical
    /// block descriptors remain distinct. All offsets are arena-relative. The
    /// operation asserts residency and every capacity boundary before a write;
    /// it never truncates, resizes, or skips a semantic row.
    ///
    /// @param body Resident logical block whose read-only Phase 2 state is read.
    /// @param analyzer Analyzer-wide immutable parameters, anchor name, and
    ///                 compiled-core categories read by evaluation.
    /// @param selectedHashMemories Bit `1 << DeviceHashMemoryKind` for every
    ///                 request memory consumed by this logical block's tasks.
    /// @return Pointer-free descriptor also appended to `logicalBlocks`.
    /// @invariant The appended slices decode byte-identically to `body` and
    ///            preserve statement index order and local identifier values.
    DeviceLogicalBlockProjection Phase2ProjectionArena::appendLogicalBlock(
        const Memory& body,
        const ExpressionAnalyzer& analyzer,
        uint32_t selectedHashMemories) {
        const auto preflightStarted = std::chrono::steady_clock::now();
        assert(logicalBlocks.size() < capacity.logicalBlocks);
        assert(selectedHashMemories != 0);
        assert((selectedHashMemories & ~0x0fu) == 0);

        const auto slotsFor = [](uint32_t count) -> uint32_t {
            if (count == 0) return 0;
            uint32_t slots = 2;
            assert(count <= (std::numeric_limits<uint32_t>::max() >> 1));
            const uint32_t required = count * 2;
            while (slots < required) {
                assert(slots <= (std::numeric_limits<uint32_t>::max() >> 1));
                slots <<= 1;
            }
            return slots;
        };
        const auto byteHash = [](const char* bytes, uint32_t length) -> uint64_t {
            uint64_t value = 14695981039346656037ull;
            for (uint32_t i = 0; i < length; ++i) {
                value ^= static_cast<uint8_t>(bytes[i]);
                value *= 1099511628211ull;
            }
            return value;
        };
        const auto podHash = [](uint64_t value) -> uint64_t {
            value += 0x9e3779b97f4a7c15ull;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
            return value ^ (value >> 31);
        };
        const auto compiledCategory = [&analyzer](
            const StrSpan& value) -> DeviceCompiledCategory {
            const StrSpan core = extractExpressionSpan(value);
            if (core.len == 0) return DeviceCompiledCategory::absent;
            const LogicalEntity* entity = analyzer.compiledEntity(core);
            if (entity == nullptr) return DeviceCompiledCategory::absent;
            return equalSpans(StrSpan(entity->category), StrSpan("atomic", 6))
                ? DeviceCompiledCategory::atomic
                : DeviceCompiledCategory::nonAtomic;
        };

        const int32_t statementCountSigned = body.intEncodedStatementsCount();
        assert(statementCountSigned >= 0);
        const std::size_t statementCount =
            static_cast<std::size_t>(statementCountSigned);
        assert(statements.size() + statementCount <= capacity.statements);

        const int32_t nameCountSigned = body.nameMap.nameCount();
        assert(nameCountSigned >= 0);
        const std::size_t localNameRecordCount =
            static_cast<std::size_t>(nameCountSigned) + 1;
        const int64_t bodyNameBytes =
            body.nameMap.names->keyStore().logicalByteCount();
        assert(bodyNameBytes >= 0);
        const SharedNameProjection* sharedName = nullptr;
        for (const SharedNameProjection& candidate
             : sharedNameProjections_) {
            assert(candidate.source != nullptr);
            const Memory& source = *candidate.source;
            if (source.nameMap.nameCount() != nameCountSigned) continue;
            if (source.nameMap.names->keyStore().logicalByteCount()
                != bodyNameBytes) continue;
            bool equal = true;
            for (NameId id = 1; id <= nameCountSigned; ++id) {
                if (source.nameMap.parentOf(id) != body.nameMap.parentOf(id)
                    || !equalSpans(source.nameMap.decodeView(id),
                                   body.nameMap.decodeView(id))) {
                    equal = false;
                    break;
                }
            }
            if (equal) {
                sharedName = &candidate;
                break;
            }
        }
        const std::size_t localNameBytes = sharedName == nullptr
            ? static_cast<std::size_t>(bodyNameBytes)
            : 0;
        if (sharedName == nullptr) {
            assert(nameRecords.size() + localNameRecordCount
                <= capacity.nameRecords);
            assert(nameBytes.size() + localNameBytes <= capacity.nameBytes);
        }
        assert(statements.size() <= std::numeric_limits<uint32_t>::max());
        assert(nameRecords.size() <= std::numeric_limits<uint32_t>::max());
        assert(nameBytes.size() <= std::numeric_limits<uint32_t>::max());
        assert(statementCount <= std::numeric_limits<uint32_t>::max());
        assert(localNameRecordCount <= std::numeric_limits<uint32_t>::max());
        assert(localNameBytes <= std::numeric_limits<uint32_t>::max());

        DeviceLogicalBlockProjection descriptor{};
        descriptor.statementOffset = static_cast<uint32_t>(statements.size());
        descriptor.statementCount = static_cast<uint32_t>(statementCount);
        descriptor.nameRecordOffset = sharedName == nullptr
            ? static_cast<uint32_t>(nameRecords.size())
            : sharedName->recordOffset;
        descriptor.nameRecordCount = sharedName == nullptr
            ? static_cast<uint32_t>(localNameRecordCount)
            : sharedName->recordCount;
        descriptor.nameByteOffset = sharedName == nullptr
            ? static_cast<uint32_t>(nameBytes.size())
            : sharedName->byteOffset;
        descriptor.nameByteCount = sharedName == nullptr
            ? static_cast<uint32_t>(localNameBytes)
            : sharedName->byteCount;

        const auto preflightFinished = std::chrono::steady_clock::now();

        for (int32_t index = 0; index < statementCountSigned; ++index)
            statements.push_back(body.intEncodedStatements[index]);

        const auto statementsFinished = std::chrono::steady_clock::now();

        std::chrono::steady_clock::time_point nameRecordsFinished;
        std::chrono::steady_clock::time_point nameSortFinished;
        std::chrono::steady_clock::time_point nameRanksFinished;
        std::chrono::steady_clock::time_point namesFinished;
        if (sharedName == nullptr) {
        nameRecords.push_back(DeviceNameRecord{
            descriptor.nameByteOffset, 0, 0,
            std::numeric_limits<uint32_t>::max(),
            DeviceCompiledCategory::absent });
        decodedRankScratch_.clear();
        for (NameId id = 1; id <= nameCountSigned; ++id) {
            const StrSpan decoded = body.nameMap.decodeView(id);
            assert(nameBytes.size() <= std::numeric_limits<uint32_t>::max());
            const uint32_t byteOffset = static_cast<uint32_t>(nameBytes.size());
            nameBytes.insert(nameBytes.end(), decoded.ptr, decoded.ptr + decoded.len);
            nameRecords.push_back(DeviceNameRecord{
                byteOffset,
                static_cast<uint32_t>(decoded.len),
                body.nameMap.parentOf(id),
                0,
                compiledCategory(decoded) });
            decodedRankScratch_.push_back(id);
        }

        nameRecordsFinished = std::chrono::steady_clock::now();

        decodedRankWorkScratch_.resize(decodedRankScratch_.size());
        decodedRankRangeScratch_.clear();
        if (decodedRankScratch_.size() > 1) {
            assert(decodedRankScratch_.size()
                <= std::numeric_limits<uint32_t>::max());
            decodedRankRangeScratch_.push_back(DecodedRankRange{
                0,
                static_cast<uint32_t>(decodedRankScratch_.size()),
                0 });
        }
        while (!decodedRankRangeScratch_.empty()) {
            const DecodedRankRange range = decodedRankRangeScratch_.back();
            decodedRankRangeScratch_.pop_back();
            assert(range.begin < range.end);
            assert(range.end <= decodedRankScratch_.size());

            const NameId firstId = decodedRankScratch_[range.begin];
            const DeviceNameRecord& firstRecord = nameRecords[
                descriptor.nameRecordOffset
                + static_cast<uint32_t>(firstId)];
            uint32_t splitDepth = firstRecord.byteLength;
            for (uint32_t index = range.begin + 1;
                 index < range.end && splitDepth > range.depth; ++index) {
                const NameId id = decodedRankScratch_[index];
                const DeviceNameRecord& record = nameRecords[
                    descriptor.nameRecordOffset + static_cast<uint32_t>(id)];
                const uint32_t comparisonEnd =
                    std::min(splitDepth, record.byteLength);
                uint32_t cursor = range.depth;
                while (cursor < comparisonEnd
                       && nameBytes[firstRecord.byteOffset + cursor]
                           == nameBytes[record.byteOffset + cursor]) {
                    ++cursor;
                }
                splitDepth = cursor;
            }

            std::array<uint32_t, 257> counts{};
            for (uint32_t index = range.begin; index < range.end; ++index) {
                const NameId id = decodedRankScratch_[index];
                const DeviceNameRecord& record = nameRecords[
                    descriptor.nameRecordOffset + static_cast<uint32_t>(id)];
                const uint32_t bucket = splitDepth >= record.byteLength
                    ? 0u
                    : static_cast<uint32_t>(static_cast<unsigned char>(
                          nameBytes[record.byteOffset + splitDepth])) + 1u;
                assert(counts[bucket] < std::numeric_limits<uint32_t>::max());
                ++counts[bucket];
            }

            std::array<uint32_t, 257> starts{};
            std::array<uint32_t, 257> cursors{};
            uint32_t next = range.begin;
            uint32_t occupiedBuckets = 0;
            for (uint32_t bucket = 0; bucket < counts.size(); ++bucket) {
                if (counts[bucket] > 0) ++occupiedBuckets;
                starts[bucket] = next;
                cursors[bucket] = next;
                assert(next <= range.end - counts[bucket]);
                next += counts[bucket];
            }
            assert(next == range.end);
            assert(occupiedBuckets > 1
                && "distinct projected names must split after their common prefix");

            for (uint32_t index = range.begin; index < range.end; ++index) {
                const NameId id = decodedRankScratch_[index];
                const DeviceNameRecord& record = nameRecords[
                    descriptor.nameRecordOffset + static_cast<uint32_t>(id)];
                const uint32_t bucket = splitDepth >= record.byteLength
                    ? 0u
                    : static_cast<uint32_t>(static_cast<unsigned char>(
                          nameBytes[record.byteOffset + splitDepth])) + 1u;
                assert(cursors[bucket] < starts[bucket] + counts[bucket]);
                decodedRankWorkScratch_[cursors[bucket]++] = id;
            }
            std::copy(decodedRankWorkScratch_.begin() + range.begin,
                      decodedRankWorkScratch_.begin() + range.end,
                      decodedRankScratch_.begin() + range.begin);

            for (uint32_t bucket = 1; bucket < counts.size(); ++bucket) {
                if (counts[bucket] <= 1) continue;
                assert(splitDepth < std::numeric_limits<uint32_t>::max());
                assert(decodedRankRangeScratch_.size()
                    < decodedRankRangeScratch_.capacity());
                decodedRankRangeScratch_.push_back(DecodedRankRange{
                    starts[bucket], starts[bucket] + counts[bucket],
                    splitDepth + 1 });
            }
        }
        nameSortFinished = std::chrono::steady_clock::now();
        for (uint32_t rank = 0;
             rank < static_cast<uint32_t>(decodedRankScratch_.size());
             ++rank) {
            const NameId id = decodedRankScratch_[rank];
            nameRecords[descriptor.nameRecordOffset
                + static_cast<uint32_t>(id)].decodedLexRank = rank;
        }

        nameRanksFinished = std::chrono::steady_clock::now();

        descriptor.nameSlotOffset = static_cast<uint32_t>(nameSlots.size());
        descriptor.nameSlotCount = slotsFor(
            static_cast<uint32_t>(nameCountSigned));
        assert(nameSlots.size() + descriptor.nameSlotCount <= capacity.nameSlots);
        nameSlots.resize(nameSlots.size() + descriptor.nameSlotCount, -1);
        for (NameId id = 1; id <= nameCountSigned; ++id) {
            const uint32_t recordIndex = descriptor.nameRecordOffset
                + static_cast<uint32_t>(id);
            const DeviceNameRecord& record = nameRecords[recordIndex];
            uint32_t slot = static_cast<uint32_t>(byteHash(
                nameBytes.data() + record.byteOffset, record.byteLength))
                & (descriptor.nameSlotCount - 1);
            while (nameSlots[descriptor.nameSlotOffset + slot] != -1)
                slot = (slot + 1) & (descriptor.nameSlotCount - 1);
            nameSlots[descriptor.nameSlotOffset + slot] =
                static_cast<int32_t>(recordIndex);
        }

        namesFinished = std::chrono::steady_clock::now();
        sharedNameProjections_.push_back(SharedNameProjection{
            &body,
            descriptor.nameRecordOffset,
            descriptor.nameRecordCount,
            descriptor.nameByteOffset,
            descriptor.nameByteCount,
            descriptor.nameSlotOffset,
            descriptor.nameSlotCount });
        }
        else {
            descriptor.nameSlotOffset = sharedName->slotOffset;
            descriptor.nameSlotCount = sharedName->slotCount;
            const auto sharedNamesFinished =
                std::chrono::steady_clock::now();
            nameRecordsFinished = sharedNamesFinished;
            nameSortFinished = sharedNamesFinished;
            nameRanksFinished = sharedNamesFinished;
            namesFinished = sharedNamesFinished;
        }

        const int32_t ruleStringCountSigned = body.ruleInterner.internedCount();
        assert(ruleStringCountSigned >= 0);
        const uint32_t localRuleStringRecordCount =
            static_cast<uint32_t>(ruleStringCountSigned) + 1;
        assert(body.ruleInterner.table != nullptr);
        const int64_t bodyRuleStringBytes =
            body.ruleInterner.table->keyStore().logicalByteCount();
        assert(bodyRuleStringBytes >= 0);
        const SharedRuleStringProjection* sharedRuleStrings = nullptr;
        for (const SharedRuleStringProjection& candidate
             : sharedRuleStringProjections_) {
            assert(candidate.source != nullptr);
            const Memory& source = *candidate.source;
            if (source.ruleInterner.internedCount()
                != ruleStringCountSigned) continue;
            assert(source.ruleInterner.table != nullptr);
            if (source.ruleInterner.table->keyStore().logicalByteCount()
                != bodyRuleStringBytes) continue;
            bool equal = true;
            for (int32_t id = 1; id <= ruleStringCountSigned; ++id) {
                if (!equalSpans(source.ruleInterner.decodeView(id),
                                body.ruleInterner.decodeView(id))) {
                    equal = false;
                    break;
                }
            }
            if (equal) {
                sharedRuleStrings = &candidate;
                break;
            }
        }
        if (sharedRuleStrings == nullptr) {
            assert(ruleStringRecords.size() + localRuleStringRecordCount
                <= capacity.ruleStringRecords);
            assert(ruleStringBytes.size()
                    + static_cast<std::size_t>(bodyRuleStringBytes)
                <= capacity.ruleStringBytes);
        }
        descriptor.ruleStringRecordOffset = sharedRuleStrings == nullptr
            ? static_cast<uint32_t>(ruleStringRecords.size())
            : sharedRuleStrings->recordOffset;
        descriptor.ruleStringRecordCount = sharedRuleStrings == nullptr
            ? localRuleStringRecordCount
            : sharedRuleStrings->recordCount;
        descriptor.ruleStringByteOffset = sharedRuleStrings == nullptr
            ? static_cast<uint32_t>(ruleStringBytes.size())
            : sharedRuleStrings->byteOffset;
        descriptor.ruleStringByteCount = sharedRuleStrings == nullptr
            ? static_cast<uint32_t>(bodyRuleStringBytes)
            : sharedRuleStrings->byteCount;
        std::chrono::steady_clock::time_point ruleStringsFinished;
        if (sharedRuleStrings == nullptr) {
        ruleStringRecords.push_back(DeviceRuleStringRecord{
            descriptor.ruleStringByteOffset, 0,
            DeviceCompiledCategory::absent });
        for (int32_t id = 1; id <= ruleStringCountSigned; ++id) {
            const StrSpan decoded = body.ruleInterner.decodeView(id);
            assert(decoded.len >= 0);
            assert(ruleStringBytes.size() + static_cast<std::size_t>(decoded.len)
                <= capacity.ruleStringBytes);
            assert(ruleStringBytes.size()
                <= std::numeric_limits<uint32_t>::max());
            const uint32_t byteOffset =
                static_cast<uint32_t>(ruleStringBytes.size());
            ruleStringBytes.insert(
                ruleStringBytes.end(), decoded.ptr, decoded.ptr + decoded.len);
            ruleStringRecords.push_back(DeviceRuleStringRecord{
                byteOffset, static_cast<uint32_t>(decoded.len),
                compiledCategory(decoded) });
        }
        assert(static_cast<uint32_t>(ruleStringBytes.size())
                - descriptor.ruleStringByteOffset
            == descriptor.ruleStringByteCount);
        ruleStringsFinished = std::chrono::steady_clock::now();
        sharedRuleStringProjections_.push_back(SharedRuleStringProjection{
            &body,
            descriptor.ruleStringRecordOffset,
            descriptor.ruleStringRecordCount,
            descriptor.ruleStringByteOffset,
            descriptor.ruleStringByteCount });
        }
        else {
            ruleStringsFinished = std::chrono::steady_clock::now();
        }

        const auto insertByteMapSlot = [&](const DeviceByteMapView& view,
                                            uint32_t entryIndex) {
            const DeviceByteMapEntry& entry = byteMapEntries[entryIndex];
            uint32_t slot = static_cast<uint32_t>(byteHash(
                byteKeyBytes.data() + entry.keyOffset, entry.keyLength))
                & (view.slotCount - 1);
            while (byteMapSlots[view.slotOffset + slot] != -1)
                slot = (slot + 1) & (view.slotCount - 1);
            byteMapSlots[view.slotOffset + slot] =
                static_cast<int32_t>(entryIndex);
        };
        const auto appendByteSet = [&](const auto& map,
                                        DeviceByteMapKind kind) -> uint32_t {
            const int32_t countSigned = map.count();
            assert(countSigned >= 0);
            const uint32_t count = static_cast<uint32_t>(countSigned);
            DeviceByteMapView view{};
            view.entryOffset = static_cast<uint32_t>(byteMapEntries.size());
            view.entryCount = count;
            view.slotOffset = static_cast<uint32_t>(byteMapSlots.size());
            view.slotCount = slotsFor(count);
            view.kind = kind;
            assert(byteMapViews.size() < capacity.byteMapViews);
            assert(byteMapEntries.size() + count <= capacity.byteMapEntries);
            assert(byteMapSlots.size() + view.slotCount <= capacity.byteMapSlots);
            byteMapSlots.resize(byteMapSlots.size() + view.slotCount, -1);
            for (int32_t id = 1; id <= countSigned; ++id) {
                const StrSpan key = map.keyAt(id);
                assert(key.len >= 0);
                assert(byteKeyBytes.size() + static_cast<std::size_t>(key.len)
                    <= capacity.byteKeyBytes);
                const uint32_t keyOffset =
                    static_cast<uint32_t>(byteKeyBytes.size());
                byteKeyBytes.insert(
                    byteKeyBytes.end(), key.ptr, key.ptr + key.len);
                byteMapEntries.push_back(DeviceByteMapEntry{
                    keyOffset, static_cast<uint32_t>(key.len),
                    static_cast<uint32_t>(blobRecords.size()), 0 });
                insertByteMapSlot(
                    view, static_cast<uint32_t>(byteMapEntries.size() - 1));
            }
            const uint32_t viewIndex = static_cast<uint32_t>(byteMapViews.size());
            byteMapViews.push_back(view);
            return viewIndex;
        };
        const auto appendByteBlobMap = [&](const auto& map,
                                            DeviceByteMapKind kind) -> uint32_t {
            const int32_t countSigned = map.count();
            assert(countSigned >= 0);
            const uint32_t count = static_cast<uint32_t>(countSigned);
            DeviceByteMapView view{};
            view.entryOffset = static_cast<uint32_t>(byteMapEntries.size());
            view.entryCount = count;
            view.slotOffset = static_cast<uint32_t>(byteMapSlots.size());
            view.slotCount = slotsFor(count);
            view.kind = kind;
            assert(byteMapViews.size() < capacity.byteMapViews);
            assert(byteMapEntries.size() + count <= capacity.byteMapEntries);
            assert(byteMapSlots.size() + view.slotCount <= capacity.byteMapSlots);
            byteMapSlots.resize(byteMapSlots.size() + view.slotCount, -1);
            for (int32_t id = 1; id <= countSigned; ++id) {
                const StrSpan key = map.keyAt(id);
                assert(key.len >= 0);
                assert(byteKeyBytes.size() + static_cast<std::size_t>(key.len)
                    <= capacity.byteKeyBytes);
                const uint32_t keyOffset =
                    static_cast<uint32_t>(byteKeyBytes.size());
                byteKeyBytes.insert(
                    byteKeyBytes.end(), key.ptr, key.ptr + key.len);
                const int32_t runLengthSigned = map.runLen(id);
                assert(runLengthSigned >= 0);
                const uint32_t runLength =
                    static_cast<uint32_t>(runLengthSigned);
                assert(blobRecords.size() + runLength <= capacity.blobRecords);
                const uint32_t recordOffset =
                    static_cast<uint32_t>(blobRecords.size());
                for (int32_t j = 0; j < runLengthSigned; ++j) {
                    const char* contiguous = nullptr;
                    int32_t lengthSigned = 0;
                    const bool isContiguous = map.inner().peekBlobAt(
                        id, j, contiguous, lengthSigned);
                    assert(lengthSigned >= 0);
                    const uint32_t length =
                        static_cast<uint32_t>(lengthSigned);
                    assert(blobBytes.size() + length <= capacity.blobBytes);
                    const uint32_t byteOffset =
                        static_cast<uint32_t>(blobBytes.size());
                    blobBytes.resize(blobBytes.size() + length);
                    if (isContiguous) {
                        std::memcpy(blobBytes.data() + byteOffset,
                                    contiguous, length);
                    }
                    else {
                        map.inner().blobAt(
                            id, j, blobBytes.data() + byteOffset);
                    }
                    blobRecords.push_back(DeviceBlobRecord{
                        byteOffset, length });
                }
                byteMapEntries.push_back(DeviceByteMapEntry{
                    keyOffset, static_cast<uint32_t>(key.len),
                    recordOffset, runLength });
                insertByteMapSlot(
                    view, static_cast<uint32_t>(byteMapEntries.size() - 1));
            }
            const uint32_t viewIndex = static_cast<uint32_t>(byteMapViews.size());
            byteMapViews.push_back(view);
            return viewIndex;
        };

        descriptor.byteMapViewOffset =
            static_cast<uint32_t>(byteMapViews.size());
        const HashMemory* const memories[4] = {
            &body.overallHashMemory,
            &body.localHashMemory,
            &body.localHashMemoryDelta,
            &body.workingMemory
        };
        const DeviceByteMapKind wholeKinds[4] = {
            DeviceByteMapKind::overallWholeKeys,
            DeviceByteMapKind::localWholeKeys,
            DeviceByteMapKind::deltaWholeKeys,
            DeviceByteMapKind::workingWholeKeys
        };
        const DeviceByteMapKind subkeyKinds[4] = {
            DeviceByteMapKind::overallSubkeys,
            DeviceByteMapKind::localSubkeys,
            DeviceByteMapKind::deltaSubkeys,
            DeviceByteMapKind::workingSubkeys
        };
        const auto appendEmptyByteMap = [&](DeviceByteMapKind kind) {
            assert(byteMapViews.size() < capacity.byteMapViews);
            DeviceByteMapView view{};
            view.entryOffset = static_cast<uint32_t>(byteMapEntries.size());
            view.slotOffset = static_cast<uint32_t>(byteMapSlots.size());
            view.kind = kind;
            byteMapViews.push_back(view);
        };
        for (uint32_t memoryIndex = 0; memoryIndex < 4; ++memoryIndex) {
            if ((selectedHashMemories & (1u << memoryIndex)) != 0) {
                appendByteSet(memories[memoryIndex]->normalizedEncodedKeys,
                    wholeKinds[memoryIndex]);
                appendByteBlobMap(memories[memoryIndex]->normalizedEncodedSubkeys,
                    subkeyKinds[memoryIndex]);
            }
            else {
                appendEmptyByteMap(wholeKinds[memoryIndex]);
                appendEmptyByteMap(subkeyKinds[memoryIndex]);
            }
        }
        appendByteBlobMap(body.overallHashMemory.encodedMap,
            DeviceByteMapKind::overallEncoded);
        const uint32_t remainingArgsViewIndex = appendByteBlobMap(
            body.overallHashMemory.remainingArgsNormalizedEncodedMap,
            DeviceByteMapKind::overallRemainingArgs);
        descriptor.byteMapViewCount =
            static_cast<uint32_t>(byteMapViews.size())
            - descriptor.byteMapViewOffset;
        assert(descriptor.byteMapViewCount == 10);

        const auto byteMapsFinished = std::chrono::steady_clock::now();

        const DeviceByteMapView& remainingView =
            byteMapViews[remainingArgsViewIndex];
        uint32_t reverseEdgeUpper = 0;
        for (uint32_t i = 0; i < remainingView.entryCount; ++i) {
            const uint32_t run = byteMapEntries[
                remainingView.entryOffset + i].blobRecordCount;
            assert(reverseEdgeUpper
                <= std::numeric_limits<uint32_t>::max() - run);
            reverseEdgeUpper += run;
        }
        DeviceReverseMapView reverseView{};
        reverseView.entryOffset = static_cast<uint32_t>(reverseMapEntries.size());
        reverseView.slotOffset = static_cast<uint32_t>(reverseMapSlots.size());
        reverseView.slotCount = slotsFor(reverseEdgeUpper);
        assert(reverseMapViews.size() < capacity.reverseMapViews);
        assert(reverseMapSlots.size() + reverseView.slotCount
            <= capacity.reverseMapSlots);
        reverseMapSlots.resize(
            reverseMapSlots.size() + reverseView.slotCount, -1);
        const auto findReverseEntry = [&](const char* key,
                                           uint32_t length) -> int32_t {
            assert(reverseView.slotCount > 0);
            uint32_t slot = static_cast<uint32_t>(byteHash(key, length))
                & (reverseView.slotCount - 1);
            while (true) {
                const int32_t entryIndex =
                    reverseMapSlots[reverseView.slotOffset + slot];
                if (entryIndex == -1) return -static_cast<int32_t>(slot) - 1;
                const DeviceReverseMapEntry& entry =
                    reverseMapEntries[static_cast<uint32_t>(entryIndex)];
                if (entry.keyLength == length
                    && std::memcmp(reverseKeyBytes.data() + entry.keyOffset,
                                   key, length) == 0) {
                    return entryIndex;
                }
                slot = (slot + 1) & (reverseView.slotCount - 1);
            }
        };
        if (reverseEdgeUpper > 0) {
            for (uint32_t i = 0; i < remainingView.entryCount; ++i) {
                const DeviceByteMapEntry& forwardEntry =
                    byteMapEntries[remainingView.entryOffset + i];
                for (uint32_t j = 0; j < forwardEntry.blobRecordCount; ++j) {
                    const DeviceBlobRecord& blob = blobRecords[
                        forwardEntry.blobRecordOffset + j];
                    const char* key = blobBytes.data() + blob.byteOffset;
                    int32_t found = findReverseEntry(key, blob.byteLength);
                    if (found < 0) {
                        const uint32_t slot =
                            static_cast<uint32_t>(-found - 1);
                        assert(reverseMapEntries.size()
                            < capacity.reverseMapEntries);
                        assert(reverseKeyBytes.size() + blob.byteLength
                            <= capacity.reverseKeyBytes);
                        const uint32_t keyOffset =
                            static_cast<uint32_t>(reverseKeyBytes.size());
                        reverseKeyBytes.insert(reverseKeyBytes.end(),
                            key, key + blob.byteLength);
                        const uint32_t entryIndex =
                            static_cast<uint32_t>(reverseMapEntries.size());
                        reverseMapEntries.push_back(DeviceReverseMapEntry{
                            keyOffset, blob.byteLength, 0, 0 });
                        reverseMapSlots[reverseView.slotOffset + slot] =
                            static_cast<int32_t>(entryIndex);
                        found = static_cast<int32_t>(entryIndex);
                    }
                    ++reverseMapEntries[static_cast<uint32_t>(found)].ownerCount;
                }
            }
        }
        reverseView.entryCount =
            static_cast<uint32_t>(reverseMapEntries.size())
            - reverseView.entryOffset;
        reverseRunScratch_.assign(reverseView.entryCount, 0);
        uint32_t ownerTotal = 0;
        for (uint32_t i = 0; i < reverseView.entryCount; ++i) {
            DeviceReverseMapEntry& entry =
                reverseMapEntries[reverseView.entryOffset + i];
            entry.ownerOffset =
                static_cast<uint32_t>(reverseOwners.size()) + ownerTotal;
            ownerTotal += entry.ownerCount;
        }
        assert(reverseOwners.size() + ownerTotal <= capacity.reverseOwners);
        reverseOwners.resize(reverseOwners.size() + ownerTotal);
        if (reverseEdgeUpper > 0) {
            for (uint32_t i = 0; i < remainingView.entryCount; ++i) {
                const DeviceByteMapEntry& forwardEntry =
                    byteMapEntries[remainingView.entryOffset + i];
                for (uint32_t j = 0; j < forwardEntry.blobRecordCount; ++j) {
                    const DeviceBlobRecord& blob = blobRecords[
                        forwardEntry.blobRecordOffset + j];
                    const int32_t found = findReverseEntry(
                        blobBytes.data() + blob.byteOffset, blob.byteLength);
                    assert(found >= 0);
                    DeviceReverseMapEntry& entry =
                        reverseMapEntries[static_cast<uint32_t>(found)];
                    const uint32_t localEntry =
                        static_cast<uint32_t>(found) - reverseView.entryOffset;
                    const uint32_t cursor = reverseRunScratch_[localEntry]++;
                    assert(cursor < entry.ownerCount);
                    reverseOwners[entry.ownerOffset + cursor] =
                        static_cast<int32_t>(i + 1);
                }
            }
        }
        descriptor.reverseMapViewOffset =
            static_cast<uint32_t>(reverseMapViews.size());
        descriptor.reverseMapViewCount = 1;
        reverseMapViews.push_back(reverseView);

        const auto reverseMapFinished = std::chrono::steady_clock::now();

        const auto insertPodMapSlot = [&](const DevicePodMapView& view,
                                           uint32_t entryIndex) {
            const DevicePodMapEntry& entry = podMapEntries[entryIndex];
            uint32_t slot = static_cast<uint32_t>(podHash(
                static_cast<uint64_t>(entry.key))) & (view.slotCount - 1);
            while (podMapSlots[view.slotOffset + slot] != -1)
                slot = (slot + 1) & (view.slotCount - 1);
            podMapSlots[view.slotOffset + slot] =
                static_cast<int32_t>(entryIndex);
        };
        const auto appendPodSet = [&](const auto& map, DevicePodMapKind kind) {
            const int32_t countSigned = map.count();
            assert(countSigned >= 0);
            const uint32_t count = static_cast<uint32_t>(countSigned);
            DevicePodMapView view{};
            view.entryOffset = static_cast<uint32_t>(podMapEntries.size());
            view.entryCount = count;
            view.slotOffset = static_cast<uint32_t>(podMapSlots.size());
            view.slotCount = slotsFor(count);
            view.kind = kind;
            assert(podMapViews.size() < capacity.podMapViews);
            assert(podMapEntries.size() + count <= capacity.podMapEntries);
            assert(podMapSlots.size() + view.slotCount <= capacity.podMapSlots);
            podMapSlots.resize(podMapSlots.size() + view.slotCount, -1);
            for (int32_t id = 1; id <= countSigned; ++id) {
                podMapEntries.push_back(DevicePodMapEntry{
                    static_cast<int64_t>(map.keyAt(id)), 0,
                    static_cast<uint32_t>(podRunValues.size()), 0 });
                insertPodMapSlot(
                    view, static_cast<uint32_t>(podMapEntries.size() - 1));
            }
            podMapViews.push_back(view);
        };
        const auto appendPodRunMap = [&](const auto& map,
                                          DevicePodMapKind kind) {
            const int32_t countSigned = map.count();
            assert(countSigned >= 0);
            const uint32_t count = static_cast<uint32_t>(countSigned);
            DevicePodMapView view{};
            view.entryOffset = static_cast<uint32_t>(podMapEntries.size());
            view.entryCount = count;
            view.slotOffset = static_cast<uint32_t>(podMapSlots.size());
            view.slotCount = slotsFor(count);
            view.kind = kind;
            assert(podMapViews.size() < capacity.podMapViews);
            assert(podMapEntries.size() + count <= capacity.podMapEntries);
            assert(podMapSlots.size() + view.slotCount <= capacity.podMapSlots);
            podMapSlots.resize(podMapSlots.size() + view.slotCount, -1);
            for (int32_t id = 1; id <= countSigned; ++id) {
                const int32_t runLengthSigned = map.runLen(id);
                assert(runLengthSigned >= 0);
                const uint32_t runLength =
                    static_cast<uint32_t>(runLengthSigned);
                assert(podRunValues.size() + runLength
                    <= capacity.podRunValues);
                const uint32_t runOffset =
                    static_cast<uint32_t>(podRunValues.size());
                for (int32_t j = 0; j < runLengthSigned; ++j)
                    podRunValues.push_back(map.valueAt(id, j));
                podMapEntries.push_back(DevicePodMapEntry{
                    static_cast<int64_t>(map.keyAt(id)), 0,
                    runOffset, runLength });
                insertPodMapSlot(
                    view, static_cast<uint32_t>(podMapEntries.size() - 1));
            }
            podMapViews.push_back(view);
        };
        const auto appendStatementVectorSet = [&](const auto& statements,
                                                   DevicePodMapKind kind) {
            const int32_t countSigned = statements.size();
            assert(countSigned >= 0);
            const uint32_t count = static_cast<uint32_t>(countSigned);
            DevicePodMapView view{};
            view.entryOffset = static_cast<uint32_t>(podMapEntries.size());
            view.entryCount = count;
            view.slotOffset = static_cast<uint32_t>(podMapSlots.size());
            view.slotCount = slotsFor(count);
            view.kind = kind;
            assert(podMapViews.size() < capacity.podMapViews);
            assert(podMapEntries.size() + count <= capacity.podMapEntries);
            assert(podMapSlots.size() + view.slotCount <= capacity.podMapSlots);
            podMapSlots.resize(podMapSlots.size() + view.slotCount, -1);
            for (int32_t index = 0; index < countSigned; ++index) {
                const IntEncodedExpr& statement = statements[index];
                podMapEntries.push_back(DevicePodMapEntry{
                    packStatementKey(
                        statement.originalId, statement.validityId),
                    0, static_cast<uint32_t>(podRunValues.size()), 0 });
                insertPodMapSlot(
                    view, static_cast<uint32_t>(podMapEntries.size() - 1));
            }
            podMapViews.push_back(view);
        };

        descriptor.podMapViewOffset =
            static_cast<uint32_t>(podMapViews.size());
        appendPodSet(body.overallHashMemory.productsOfRecursionIds,
            DevicePodMapKind::recursionProducts);
        {
            const auto& map = body.intKnownStatements;
            const int32_t countSigned = map.count();
            assert(countSigned >= 0);
            const uint32_t count = static_cast<uint32_t>(countSigned);
            DevicePodMapView view{};
            view.entryOffset = static_cast<uint32_t>(podMapEntries.size());
            view.entryCount = count;
            view.slotOffset = static_cast<uint32_t>(podMapSlots.size());
            view.slotCount = slotsFor(count);
            view.kind = DevicePodMapKind::knownStatements;
            assert(podMapViews.size() < capacity.podMapViews);
            assert(podMapEntries.size() + count <= capacity.podMapEntries);
            assert(podMapSlots.size() + view.slotCount <= capacity.podMapSlots);
            podMapSlots.resize(podMapSlots.size() + view.slotCount, -1);
            for (int32_t id = 1; id <= countSigned; ++id) {
                const StatementFlags& flags = map.valueAt(id);
                const uint64_t scalar = (flags.local ? 1ull : 0ull)
                    | (flags.fullyDisintegrated ? 2ull : 0ull);
                podMapEntries.push_back(DevicePodMapEntry{
                    static_cast<int64_t>(map.keyAt(id)), scalar,
                    static_cast<uint32_t>(podRunValues.size()), 0 });
                insertPodMapSlot(
                    view, static_cast<uint32_t>(podMapEntries.size() - 1));
            }
            podMapViews.push_back(view);
        }
        appendPodSet(body.intLocalEncodedStatementsSet,
            DevicePodMapKind::localStatements);
        appendStatementVectorSet(body.intLocalEncodedStatementsDelta,
            DevicePodMapKind::localDeltaStatements);
        appendStatementVectorSet(body.intExternalStatements,
            DevicePodMapKind::externalStatements);
        appendPodRunMap(body.intStatementLevelsMap,
            DevicePodMapKind::statementLevels);
        appendPodSet(body.intValidityNamesToFilter,
            DevicePodMapKind::validityFilter);
        appendPodSet(body.frozenOrBranches,
            DevicePodMapKind::frozenOrBranches);
        appendPodRunMap(body.intToBeProved, DevicePodMapKind::goals);
        appendPodSet(body.canBeSentIds,
            DevicePodMapKind::mailEligibleStatements);
        appendPodSet(body.canBeSentMarkerIds,
            DevicePodMapKind::mailEligibleMarkers);
        descriptor.podMapViewCount =
            static_cast<uint32_t>(podMapViews.size())
            - descriptor.podMapViewOffset;
        assert(descriptor.podMapViewCount == 11);

        const auto podMapsFinished = std::chrono::steady_clock::now();

        const auto appendMandatoryKeys = [&](const auto& source,
                                              uint32_t& offset,
                                              uint32_t& count) {
            const int32_t sourceCountSigned = source.size();
            assert(sourceCountSigned >= 0);
            const uint32_t sourceCount =
                static_cast<uint32_t>(sourceCountSigned);
            assert(mandatoryStatementKeys.size() + sourceCount
                <= capacity.mandatoryStatementKeys);
            offset = static_cast<uint32_t>(mandatoryStatementKeys.size());
            count = sourceCount;
            for (int32_t i = 0; i < sourceCountSigned; ++i) {
                mandatoryStatementKeys.push_back(packStatementKey(
                    source[i].originalId, source[i].validityId));
            }
        };
        appendMandatoryKeys(body.intLocalEncodedStatements,
            descriptor.localKeyOffset, descriptor.localKeyCount);
        appendMandatoryKeys(body.intLocalEncodedStatementsDelta,
            descriptor.deltaKeyOffset, descriptor.deltaKeyCount);
        appendMandatoryKeys(body.intExternalStatements,
            descriptor.externalKeyOffset, descriptor.externalKeyCount);

        const StrSpan expressionKey = body.exprKeyView();
        assert(expressionKey.len >= 0);
        assert(metadataBytes.size() + static_cast<std::size_t>(expressionKey.len)
            <= capacity.metadataBytes);
        descriptor.metadataOffset = static_cast<uint32_t>(metadataBytes.size());
        descriptor.metadataCount = static_cast<uint32_t>(expressionKey.len);
        metadataBytes.insert(metadataBytes.end(), expressionKey.ptr,
            expressionKey.ptr + expressionKey.len);
        const StrSpan anchorName(analyzer.anchorInfo.name);
        assert(anchorName.len >= 0);
        assert(metadataBytes.size() + static_cast<std::size_t>(anchorName.len)
            <= capacity.metadataBytes);
        descriptor.anchorNameOffset = static_cast<uint32_t>(metadataBytes.size());
        descriptor.anchorNameCount = static_cast<uint32_t>(anchorName.len);
        metadataBytes.insert(metadataBytes.end(), anchorName.ptr,
            anchorName.ptr + anchorName.len);
        descriptor.overallMaxKeyLength = body.overallHashMemory.maxKeyLength;
        descriptor.localMaxKeyLength = body.localHashMemory.maxKeyLength;
        descriptor.deltaMaxKeyLength = body.localHashMemoryDelta.maxKeyLength;
        descriptor.workingMaxKeyLength = body.workingMemory.maxKeyLength;
        descriptor.primedForContradiction =
            body.primedForContradiction ? 1 : 0;
        descriptor.isPartOfRecursion = body.isPartOfRecursion ? 1 : 0;
        descriptor.contradictionIndex = body.contradictionIndex;
        descriptor.mainValidityId = NameMap::MAIN_ID;
        descriptor.level = body.level;
        descriptor.standardMaxSecondaryNumber =
            analyzer.parameters.standardMaxSecondaryNumber;
        if (analyzer.parameters.incubator_mode)
            descriptor.evaluationFlags |= deviceEvaluationIncubatorMode;
        if (analyzer.parameters.ban_disintegration)
            descriptor.evaluationFlags |= deviceEvaluationBanDisintegration;
        if (analyzer.parameters.compressor_mode)
            descriptor.evaluationFlags |= deviceEvaluationCompressorMode;

        assert(statements.size() <= capacity.statements);
        assert(nameRecords.size() <= capacity.nameRecords);
        assert(nameBytes.size() <= capacity.nameBytes);
        logicalBlocks.push_back(descriptor);

        const auto projectionFinished = std::chrono::steady_clock::now();
        const auto addInterval = [](uint64_t& total, const auto& start,
                                    const auto& finish) {
            const int64_t elapsed = std::chrono::duration_cast<
                std::chrono::nanoseconds>(finish - start).count();
            assert(elapsed >= 0);
            assert(total <= std::numeric_limits<uint64_t>::max()
                - static_cast<uint64_t>(elapsed));
            total += static_cast<uint64_t>(elapsed);
        };
        ++timing.logicalBlocks;
        addInterval(timing.preflightNanoseconds,
            preflightStarted, preflightFinished);
        addInterval(timing.statementNanoseconds,
            preflightFinished, statementsFinished);
        addInterval(timing.nameNanoseconds,
            statementsFinished, namesFinished);
        addInterval(timing.nameRecordNanoseconds,
            statementsFinished, nameRecordsFinished);
        addInterval(timing.nameSortNanoseconds,
            nameRecordsFinished, nameSortFinished);
        addInterval(timing.nameRankNanoseconds,
            nameSortFinished, nameRanksFinished);
        addInterval(timing.nameSlotNanoseconds,
            nameRanksFinished, namesFinished);
        addInterval(timing.ruleStringNanoseconds,
            namesFinished, ruleStringsFinished);
        addInterval(timing.byteMapNanoseconds,
            ruleStringsFinished, byteMapsFinished);
        addInterval(timing.reverseMapNanoseconds,
            byteMapsFinished, reverseMapFinished);
        addInterval(timing.podMapNanoseconds,
            reverseMapFinished, podMapsFinished);
        addInterval(timing.finalNanoseconds,
            podMapsFinished, projectionFinished);
        return descriptor;
    }

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
    uint32_t Phase2ProjectionArena::appendShard(
        const Phase2ProjectionArena& source) {
        const auto mergeStarted = std::chrono::steady_clock::now();
        assert(this != &source);

        const auto checkedBase = [](std::size_t size) -> uint32_t {
            assert(size <= std::numeric_limits<uint32_t>::max());
            return static_cast<uint32_t>(size);
        };
        const uint32_t blockBase = checkedBase(logicalBlocks.size());
        const uint32_t statementBase = checkedBase(statements.size());
        const uint32_t nameRecordBase = checkedBase(nameRecords.size());
        const uint32_t nameByteBase = checkedBase(nameBytes.size());
        const uint32_t nameSlotBase = checkedBase(nameSlots.size());
        const uint32_t ruleRecordBase = checkedBase(ruleStringRecords.size());
        const uint32_t ruleByteBase = checkedBase(ruleStringBytes.size());
        const uint32_t byteViewBase = checkedBase(byteMapViews.size());
        const uint32_t byteEntryBase = checkedBase(byteMapEntries.size());
        const uint32_t byteSlotBase = checkedBase(byteMapSlots.size());
        const uint32_t byteKeyBase = checkedBase(byteKeyBytes.size());
        const uint32_t blobRecordBase = checkedBase(blobRecords.size());
        const uint32_t blobByteBase = checkedBase(blobBytes.size());
        const uint32_t reverseViewBase = checkedBase(reverseMapViews.size());
        const uint32_t reverseEntryBase = checkedBase(reverseMapEntries.size());
        const uint32_t reverseSlotBase = checkedBase(reverseMapSlots.size());
        const uint32_t reverseKeyBase = checkedBase(reverseKeyBytes.size());
        const uint32_t reverseOwnerBase = checkedBase(reverseOwners.size());
        const uint32_t podViewBase = checkedBase(podMapViews.size());
        const uint32_t podEntryBase = checkedBase(podMapEntries.size());
        const uint32_t podSlotBase = checkedBase(podMapSlots.size());
        const uint32_t podRunBase = checkedBase(podRunValues.size());
        const uint32_t mandatoryKeyBase = checkedBase(
            mandatoryStatementKeys.size());
        const uint32_t metadataBase = checkedBase(metadataBytes.size());

        assert(logicalBlocks.size() + source.logicalBlocks.size()
            <= capacity.logicalBlocks);
        assert(statements.size() + source.statements.size()
            <= capacity.statements);
        assert(nameRecords.size() + source.nameRecords.size()
            <= capacity.nameRecords);
        assert(nameBytes.size() + source.nameBytes.size() <= capacity.nameBytes);
        assert(nameSlots.size() + source.nameSlots.size() <= capacity.nameSlots);
        assert(ruleStringRecords.size() + source.ruleStringRecords.size()
            <= capacity.ruleStringRecords);
        assert(ruleStringBytes.size() + source.ruleStringBytes.size()
            <= capacity.ruleStringBytes);
        assert(byteMapViews.size() + source.byteMapViews.size()
            <= capacity.byteMapViews);
        assert(byteMapEntries.size() + source.byteMapEntries.size()
            <= capacity.byteMapEntries);
        assert(byteMapSlots.size() + source.byteMapSlots.size()
            <= capacity.byteMapSlots);
        assert(byteKeyBytes.size() + source.byteKeyBytes.size()
            <= capacity.byteKeyBytes);
        assert(blobRecords.size() + source.blobRecords.size()
            <= capacity.blobRecords);
        assert(blobBytes.size() + source.blobBytes.size() <= capacity.blobBytes);
        assert(reverseMapViews.size() + source.reverseMapViews.size()
            <= capacity.reverseMapViews);
        assert(reverseMapEntries.size() + source.reverseMapEntries.size()
            <= capacity.reverseMapEntries);
        assert(reverseMapSlots.size() + source.reverseMapSlots.size()
            <= capacity.reverseMapSlots);
        assert(reverseKeyBytes.size() + source.reverseKeyBytes.size()
            <= capacity.reverseKeyBytes);
        assert(reverseOwners.size() + source.reverseOwners.size()
            <= capacity.reverseOwners);
        assert(podMapViews.size() + source.podMapViews.size()
            <= capacity.podMapViews);
        assert(podMapEntries.size() + source.podMapEntries.size()
            <= capacity.podMapEntries);
        assert(podMapSlots.size() + source.podMapSlots.size()
            <= capacity.podMapSlots);
        assert(podRunValues.size() + source.podRunValues.size()
            <= capacity.podRunValues);
        assert(mandatoryStatementKeys.size()
                + source.mandatoryStatementKeys.size()
            <= capacity.mandatoryStatementKeys);
        assert(metadataBytes.size() + source.metadataBytes.size()
            <= capacity.metadataBytes);

        logicalBlocks.insert(logicalBlocks.end(),
            source.logicalBlocks.begin(), source.logicalBlocks.end());
        statements.insert(statements.end(),
            source.statements.begin(), source.statements.end());
        nameRecords.insert(nameRecords.end(),
            source.nameRecords.begin(), source.nameRecords.end());
        nameBytes.insert(nameBytes.end(),
            source.nameBytes.begin(), source.nameBytes.end());
        nameSlots.insert(nameSlots.end(),
            source.nameSlots.begin(), source.nameSlots.end());
        ruleStringRecords.insert(ruleStringRecords.end(),
            source.ruleStringRecords.begin(), source.ruleStringRecords.end());
        ruleStringBytes.insert(ruleStringBytes.end(),
            source.ruleStringBytes.begin(), source.ruleStringBytes.end());
        byteMapViews.insert(byteMapViews.end(),
            source.byteMapViews.begin(), source.byteMapViews.end());
        byteMapEntries.insert(byteMapEntries.end(),
            source.byteMapEntries.begin(), source.byteMapEntries.end());
        byteMapSlots.insert(byteMapSlots.end(),
            source.byteMapSlots.begin(), source.byteMapSlots.end());
        byteKeyBytes.insert(byteKeyBytes.end(),
            source.byteKeyBytes.begin(), source.byteKeyBytes.end());
        blobRecords.insert(blobRecords.end(),
            source.blobRecords.begin(), source.blobRecords.end());
        blobBytes.insert(blobBytes.end(),
            source.blobBytes.begin(), source.blobBytes.end());
        reverseMapViews.insert(reverseMapViews.end(),
            source.reverseMapViews.begin(), source.reverseMapViews.end());
        reverseMapEntries.insert(reverseMapEntries.end(),
            source.reverseMapEntries.begin(), source.reverseMapEntries.end());
        reverseMapSlots.insert(reverseMapSlots.end(),
            source.reverseMapSlots.begin(), source.reverseMapSlots.end());
        reverseKeyBytes.insert(reverseKeyBytes.end(),
            source.reverseKeyBytes.begin(), source.reverseKeyBytes.end());
        reverseOwners.insert(reverseOwners.end(),
            source.reverseOwners.begin(), source.reverseOwners.end());
        podMapViews.insert(podMapViews.end(),
            source.podMapViews.begin(), source.podMapViews.end());
        podMapEntries.insert(podMapEntries.end(),
            source.podMapEntries.begin(), source.podMapEntries.end());
        podMapSlots.insert(podMapSlots.end(),
            source.podMapSlots.begin(), source.podMapSlots.end());
        podRunValues.insert(podRunValues.end(),
            source.podRunValues.begin(), source.podRunValues.end());
        mandatoryStatementKeys.insert(mandatoryStatementKeys.end(),
            source.mandatoryStatementKeys.begin(),
            source.mandatoryStatementKeys.end());
        metadataBytes.insert(metadataBytes.end(),
            source.metadataBytes.begin(), source.metadataBytes.end());

        const auto checkedAdd = [](uint32_t value, uint32_t base) -> uint32_t {
            assert(value <= std::numeric_limits<uint32_t>::max() - base);
            return value + base;
        };
        for (uint32_t index = blockBase;
             index < logicalBlocks.size(); ++index) {
            DeviceLogicalBlockProjection& descriptor = logicalBlocks[index];
            descriptor.statementOffset = checkedAdd(
                descriptor.statementOffset, statementBase);
            descriptor.nameRecordOffset = checkedAdd(
                descriptor.nameRecordOffset, nameRecordBase);
            descriptor.nameByteOffset = checkedAdd(
                descriptor.nameByteOffset, nameByteBase);
            descriptor.nameSlotOffset = checkedAdd(
                descriptor.nameSlotOffset, nameSlotBase);
            descriptor.ruleStringRecordOffset = checkedAdd(
                descriptor.ruleStringRecordOffset, ruleRecordBase);
            descriptor.ruleStringByteOffset = checkedAdd(
                descriptor.ruleStringByteOffset, ruleByteBase);
            descriptor.byteMapViewOffset = checkedAdd(
                descriptor.byteMapViewOffset, byteViewBase);
            descriptor.reverseMapViewOffset = checkedAdd(
                descriptor.reverseMapViewOffset, reverseViewBase);
            descriptor.podMapViewOffset = checkedAdd(
                descriptor.podMapViewOffset, podViewBase);
            descriptor.localKeyOffset = checkedAdd(
                descriptor.localKeyOffset, mandatoryKeyBase);
            descriptor.deltaKeyOffset = checkedAdd(
                descriptor.deltaKeyOffset, mandatoryKeyBase);
            descriptor.externalKeyOffset = checkedAdd(
                descriptor.externalKeyOffset, mandatoryKeyBase);
            descriptor.metadataOffset = checkedAdd(
                descriptor.metadataOffset, metadataBase);
            descriptor.anchorNameOffset = checkedAdd(
                descriptor.anchorNameOffset, metadataBase);
        }
        for (uint32_t index = nameRecordBase;
             index < nameRecords.size(); ++index)
            nameRecords[index].byteOffset = checkedAdd(
                nameRecords[index].byteOffset, nameByteBase);
        for (uint32_t index = nameSlotBase; index < nameSlots.size(); ++index)
            if (nameSlots[index] >= 0)
                nameSlots[index] += static_cast<int32_t>(nameRecordBase);
        for (uint32_t index = ruleRecordBase;
             index < ruleStringRecords.size(); ++index)
            ruleStringRecords[index].byteOffset = checkedAdd(
                ruleStringRecords[index].byteOffset, ruleByteBase);
        for (uint32_t index = byteViewBase;
             index < byteMapViews.size(); ++index) {
            byteMapViews[index].entryOffset = checkedAdd(
                byteMapViews[index].entryOffset, byteEntryBase);
            byteMapViews[index].slotOffset = checkedAdd(
                byteMapViews[index].slotOffset, byteSlotBase);
        }
        for (uint32_t index = byteEntryBase;
             index < byteMapEntries.size(); ++index) {
            byteMapEntries[index].keyOffset = checkedAdd(
                byteMapEntries[index].keyOffset, byteKeyBase);
            byteMapEntries[index].blobRecordOffset = checkedAdd(
                byteMapEntries[index].blobRecordOffset, blobRecordBase);
        }
        for (uint32_t index = byteSlotBase;
             index < byteMapSlots.size(); ++index)
            if (byteMapSlots[index] >= 0)
                byteMapSlots[index] += static_cast<int32_t>(byteEntryBase);
        for (uint32_t index = blobRecordBase;
             index < blobRecords.size(); ++index)
            blobRecords[index].byteOffset = checkedAdd(
                blobRecords[index].byteOffset, blobByteBase);
        for (uint32_t index = reverseViewBase;
             index < reverseMapViews.size(); ++index) {
            reverseMapViews[index].entryOffset = checkedAdd(
                reverseMapViews[index].entryOffset, reverseEntryBase);
            reverseMapViews[index].slotOffset = checkedAdd(
                reverseMapViews[index].slotOffset, reverseSlotBase);
        }
        for (uint32_t index = reverseEntryBase;
             index < reverseMapEntries.size(); ++index) {
            reverseMapEntries[index].keyOffset = checkedAdd(
                reverseMapEntries[index].keyOffset, reverseKeyBase);
            reverseMapEntries[index].ownerOffset = checkedAdd(
                reverseMapEntries[index].ownerOffset, reverseOwnerBase);
        }
        for (uint32_t index = reverseSlotBase;
             index < reverseMapSlots.size(); ++index)
            if (reverseMapSlots[index] >= 0)
                reverseMapSlots[index] += static_cast<int32_t>(reverseEntryBase);
        for (uint32_t index = podViewBase;
             index < podMapViews.size(); ++index) {
            podMapViews[index].entryOffset = checkedAdd(
                podMapViews[index].entryOffset, podEntryBase);
            podMapViews[index].slotOffset = checkedAdd(
                podMapViews[index].slotOffset, podSlotBase);
        }
        for (uint32_t index = podEntryBase;
             index < podMapEntries.size(); ++index)
            podMapEntries[index].runOffset = checkedAdd(
                podMapEntries[index].runOffset, podRunBase);
        for (uint32_t index = podSlotBase;
             index < podMapSlots.size(); ++index)
            if (podMapSlots[index] >= 0)
                podMapSlots[index] += static_cast<int32_t>(podEntryBase);

        const auto addCounter = [](uint64_t& destination, uint64_t value) {
            assert(destination <= std::numeric_limits<uint64_t>::max() - value);
            destination += value;
        };
        addCounter(timing.preflightNanoseconds,
            source.timing.preflightNanoseconds);
        addCounter(timing.statementNanoseconds,
            source.timing.statementNanoseconds);
        addCounter(timing.nameNanoseconds, source.timing.nameNanoseconds);
        addCounter(timing.nameRecordNanoseconds,
            source.timing.nameRecordNanoseconds);
        addCounter(timing.nameSortNanoseconds,
            source.timing.nameSortNanoseconds);
        addCounter(timing.nameRankNanoseconds,
            source.timing.nameRankNanoseconds);
        addCounter(timing.nameSlotNanoseconds,
            source.timing.nameSlotNanoseconds);
        addCounter(timing.ruleStringNanoseconds,
            source.timing.ruleStringNanoseconds);
        addCounter(timing.byteMapNanoseconds,
            source.timing.byteMapNanoseconds);
        addCounter(timing.reverseMapNanoseconds,
            source.timing.reverseMapNanoseconds);
        addCounter(timing.podMapNanoseconds,
            source.timing.podMapNanoseconds);
        addCounter(timing.finalNanoseconds,
            source.timing.finalNanoseconds);
        timing.logicalBlocks = logicalBlocks.size();

        const int64_t elapsed = std::chrono::duration_cast<
            std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - mergeStarted).count();
        assert(elapsed >= 0);
        addCounter(timing.mergeNanoseconds, static_cast<uint64_t>(elapsed));
        return blockBase;
    }

    /// @brief Reserve every task-projection array to a fixed ceiling.
    ///
    /// @details
    /// Validates positive capacities, reserves all four arrays once, and leaves
    /// their used prefixes empty for the first task batch.
    ///
    /// @param fixedCapacity Immutable element ceilings for this arena lifetime.
    /// @return An empty task arena owning all host reservations.
    /// @invariant No later method changes an array's allocation capacity.
    Phase2TaskProjectionArena::Phase2TaskProjectionArena(
        Phase2TaskProjectionCapacity fixedCapacity)
        : capacity(fixedCapacity) {
        assert(capacity.tasks > 0);
        assert(capacity.batches > 0);
        assert(capacity.terms > 0);
        assert(capacity.stumps > 0);
        tasks.reserve(capacity.tasks);
        batches.reserve(capacity.batches);
        terms.reserve(capacity.terms);
        stumps.reserve(capacity.stumps);
    }

    /// @brief Reset task used prefixes while retaining fixed reservations.
    ///
    /// @details
    /// Clears tasks, batches, mandatory terms, and expression stumps without
    /// shrinking or replacing any vector.
    ///
    /// @return Nothing.
    /// @invariant Every used prefix is empty and every capacity is unchanged.
    void Phase2TaskProjectionArena::clear() {
        tasks.clear();
        batches.clear();
        terms.clear();
        stumps.clear();
        assert(tasks.capacity() >= capacity.tasks);
        assert(batches.capacity() >= capacity.batches);
        assert(terms.capacity() >= capacity.terms);
        assert(stumps.capacity() >= capacity.stumps);
    }

    /// @brief Append one executor part and all of its request-local inputs.
    ///
    /// @details
    /// Copies the ordered request-batch list and its ordered mandatory terms,
    /// then copies the split stump bucket exactly. An unsplit part has no stumps
    /// and zero ordinal/total; a split part has a non-empty stump run and a valid
    /// sibling position. All references become arena-relative offsets. Every
    /// boundary and bivalent marker asserts before write.
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
    DevicePhase2Task Phase2TaskProjectionArena::appendTask(
        uint32_t logicalBlockIndex,
        const Phase2RequestBatchInput* batchInputs,
        uint32_t batchCount,
        const ExpressionStump* sourceStumps,
        uint32_t stumpCount,
        NameId stumpOrdinal,
        NameId stumpTotal,
        int32_t maximumIterationNumberVariable,
        uint32_t counterExampleMode) {
        assert(tasks.size() < capacity.tasks);
        assert(batchInputs != nullptr);
        assert(batchCount > 0);
        assert(batches.size() + batchCount <= capacity.batches);
        assert(counterExampleMode <= 1);
        assert(maximumIterationNumberVariable >= 0);
        assert((stumpCount == 0) == (sourceStumps == nullptr));
        if (stumpCount == 0) {
            assert(stumpOrdinal == 0);
            assert(stumpTotal == 0);
        }
        else {
            assert(stumpOrdinal >= 0);
            assert(stumpTotal > 0);
            assert(stumpOrdinal < stumpTotal);
        }
        assert(stumps.size() + stumpCount <= capacity.stumps);

        DevicePhase2Task task{};
        task.logicalBlockIndex = logicalBlockIndex;
        task.batchOffset = static_cast<uint32_t>(batches.size());
        task.batchCount = batchCount;
        task.stumpOffset = static_cast<uint32_t>(stumps.size());
        task.stumpCount = stumpCount;
        task.stumpOrdinal = stumpOrdinal;
        task.stumpTotal = stumpTotal;
        task.maximumIterationNumberVariable = maximumIterationNumberVariable;
        task.counterExampleMode = counterExampleMode;

        uint32_t totalTerms = 0;
        for (uint32_t index = 0; index < batchCount; ++index) {
            const Phase2RequestBatchInput& input = batchInputs[index];
            assert(static_cast<uint32_t>(input.kind)
                <= static_cast<uint32_t>(DeviceRequestBatchKind::localDeltaRules));
            assert(static_cast<uint32_t>(input.memory)
                <= static_cast<uint32_t>(DeviceHashMemoryKind::working));
            assert(input.termCount <= 2);
            assert(totalTerms <= std::numeric_limits<uint32_t>::max()
                - input.termCount);
            totalTerms += input.termCount;
        }
        assert(terms.size() + totalTerms <= capacity.terms);

        for (uint32_t index = 0; index < batchCount; ++index) {
            const Phase2RequestBatchInput& input = batchInputs[index];
            DeviceRequestBatch batch{};
            batch.kind = input.kind;
            batch.memory = input.memory;
            batch.termOffset = static_cast<uint32_t>(terms.size());
            batch.termCount = input.termCount;
            for (uint32_t termIndex = 0;
                 termIndex < input.termCount; ++termIndex) {
                const DeviceMandatoryTerm& term = input.terms[termIndex];
                assert(term.viewCount >= 1 && term.viewCount <= 2);
                for (uint32_t viewIndex = 0;
                     viewIndex < term.viewCount; ++viewIndex) {
                    assert(static_cast<uint32_t>(term.views[viewIndex])
                        <= static_cast<uint32_t>(
                            DeviceMandatoryViewKind::external));
                }
                terms.push_back(term);
            }
            batches.push_back(batch);
        }

        if (counterExampleMode != 0) {
            assert(batchCount == 1);
            assert(batchInputs[0].kind
                == DeviceRequestBatchKind::counterExample);
            assert(batchInputs[0].memory == DeviceHashMemoryKind::overall);
            assert(batchInputs[0].termCount == 0);
            assert(stumpCount == 0);
        }
        else {
            for (uint32_t index = 0; index < batchCount; ++index)
                assert(batchInputs[index].kind
                    != DeviceRequestBatchKind::counterExample);
        }

        for (uint32_t index = 0; index < stumpCount; ++index) {
            const ExpressionStump& source = sourceStumps[index];
            assert(source.count > 0);
            assert(source.count <= ExecutionParameters::MAX_EXPRESSIONS);
            assert(source.terminalOnly <= 1);
            DeviceExpressionStump stump{};
            stump.count = source.count;
            stump.terminalOnly = source.terminalOnly;
            for (NameId statement = 0; statement < source.count; ++statement) {
                assert(source.allIdx[statement] >= 0);
                stump.statementIndices[statement] = source.allIdx[statement];
            }
            stumps.push_back(stump);
        }

        tasks.push_back(task);
        return task;
    }

    /// @brief Reserve the global filter-call schedule once.
    ///
    /// @details
    /// Requires positive call, row, and per-call ceilings, reserves original and
    /// class descriptor columns plus mappings and multiplicities, and constructs
    /// one fixed power-of-two open-address table at no more than one-half load.
    ///
    /// @param fixedCapacity Immutable schedule and result ceilings.
    /// @return An empty reusable schedule.
    /// @invariant No later method changes any vector capacity.
    Phase2FilterScheduleArena::Phase2FilterScheduleArena(
        Phase2FilterScheduleCapacity fixedCapacity)
        : capacity(fixedCapacity) {
        assert(capacity.calls > 0);
        assert(capacity.examinedRows > 0);
        assert(capacity.retainedRows > 0);
        assert(capacity.maximumExaminedRowsPerCall > 0);
        assert(capacity.calls
            <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
        assert(capacity.calls <= (1u << (64u - kDeviceFilterCallShift)));
        assert(capacity.maximumExaminedRowsPerCall
            <= (1u << kDeviceFilterStatementIndexBits));
        assert(capacity.calls <= std::numeric_limits<uint32_t>::max() / 2u);
        calls.reserve(capacity.calls);
        classes.reserve(capacity.calls);
        callClassIndices.reserve(capacity.calls);
        classMultiplicities.reserve(capacity.calls);
        uint32_t slotCapacity = 1;
        while (slotCapacity < capacity.calls * 2u) {
            assert(slotCapacity <= std::numeric_limits<uint32_t>::max() / 2u);
            slotCapacity <<= 1u;
        }
        classSlots.assign(slotCapacity, -1);
    }

    /// @brief Reset the global filter schedule without releasing storage.
    ///
    /// @details
    /// Clears original and class descriptor prefixes, mappings, multiplicities,
    /// both row accumulators, and every hash slot while retaining all storage.
    ///
    /// @return Nothing.
    /// @invariant Every allocation and declared ceiling is retained.
    void Phase2FilterScheduleArena::clear() {
        calls.clear();
        classes.clear();
        callClassIndices.clear();
        classMultiplicities.clear();
        std::fill(classSlots.begin(), classSlots.end(), -1);
        examinedRows = 0;
        classExaminedRows = 0;
        assert(calls.capacity() >= capacity.calls);
        assert(classes.capacity() >= capacity.calls);
        assert(callClassIndices.capacity() >= capacity.calls);
        assert(classMultiplicities.capacity() >= capacity.calls);
    }

    /// @brief Append one processor-order filter call to the bulk schedule.
    ///
    /// @details
    /// Validates every enumeration and scalar field, accounts the complete input
    /// slice, interns the exact five-field descriptor with fixed open addressing,
    /// and appends its class index beside the processor-order call. No statement
    /// or map bytes are copied because CUDA reads the resident projection.
    ///
    /// @param logicalBlockIndex Resident logical-block projection index.
    /// @param memory Hash-memory registry supplying whole and subkey maps.
    /// @param maximumIterationNumberVariable Inclusive iteration ceiling.
    /// @param alsoAcceptFullKeys Bivalent processor whole-key widening flag.
    /// @param statementCount Full resident statement slice examined by the call.
    /// @return Pointer-free descriptor also appended to `calls`.
    /// @invariant Call order exactly matches the processor schedule; class order
    ///            is first occurrence; neither row total exceeds fixed ceilings.
    DevicePhase2FilterCall Phase2FilterScheduleArena::appendCall(
        uint32_t logicalBlockIndex,
        DeviceHashMemoryKind memory,
        int32_t maximumIterationNumberVariable,
        uint32_t alsoAcceptFullKeys,
        uint32_t statementCount) {
        assert(calls.size() < capacity.calls);
        assert(static_cast<uint32_t>(memory)
            <= static_cast<uint32_t>(DeviceHashMemoryKind::working));
        assert(maximumIterationNumberVariable >= 0);
        assert(alsoAcceptFullKeys <= 1);
        assert(statementCount <= capacity.maximumExaminedRowsPerCall);
        assert(statementCount <= capacity.examinedRows - examinedRows);

        DevicePhase2FilterCall call{};
        call.logicalBlockIndex = logicalBlockIndex;
        call.memory = memory;
        call.maximumIterationNumberVariable = maximumIterationNumberVariable;
        call.alsoAcceptFullKeys = alsoAcceptFullKeys;
        call.statementCount = statementCount;

        uint64_t signature = 1469598103934665603ull;
        const auto mix = [&signature](uint32_t value) {
            signature ^= value;
            signature *= 1099511628211ull;
        };
        mix(call.logicalBlockIndex);
        mix(static_cast<uint32_t>(call.memory));
        mix(static_cast<uint32_t>(call.maximumIterationNumberVariable));
        mix(call.alsoAcceptFullKeys);
        mix(call.statementCount);
        assert(!classSlots.empty());
        assert((classSlots.size() & (classSlots.size() - 1)) == 0);
        const uint32_t slotMask =
            static_cast<uint32_t>(classSlots.size() - 1);
        uint32_t slot = static_cast<uint32_t>(signature) & slotMask;
        uint32_t classIndex = 0;
        bool found = false;
        for (uint32_t probe = 0;
             probe < static_cast<uint32_t>(classSlots.size()); ++probe) {
            const int32_t existing = classSlots[slot];
            if (existing < 0) {
                assert(classes.size() < capacity.calls);
                classIndex = static_cast<uint32_t>(classes.size());
                classes.push_back(call);
                classMultiplicities.push_back(1);
                classSlots[slot] = static_cast<int32_t>(classIndex);
                assert(statementCount
                    <= capacity.examinedRows - classExaminedRows);
                classExaminedRows += statementCount;
                found = true;
                break;
            }
            assert(static_cast<uint32_t>(existing) < classes.size());
            const DevicePhase2FilterCall& candidate = classes[
                static_cast<uint32_t>(existing)];
            if (candidate.logicalBlockIndex == call.logicalBlockIndex
                && candidate.memory == call.memory
                && candidate.maximumIterationNumberVariable
                    == call.maximumIterationNumberVariable
                && candidate.alsoAcceptFullKeys == call.alsoAcceptFullKeys
                && candidate.statementCount == call.statementCount) {
                classIndex = static_cast<uint32_t>(existing);
                assert(classMultiplicities[classIndex]
                    < std::numeric_limits<uint32_t>::max());
                ++classMultiplicities[classIndex];
                found = true;
                break;
            }
            slot = (slot + 1u) & slotMask;
        }
        assert(found);
        calls.push_back(call);
        callClassIndices.push_back(classIndex);
        examinedRows += statementCount;
        return call;
    }

    /// @brief Measure exact filter-signature reuse without changing the route.
    ///
    /// @details
    /// Reports the production interning columns built during append. Class order
    /// is first occurrence and multiplicities retain every original call, so the
    /// scan touches only the compact multiplicity prefix and allocates no storage.
    ///
    /// @return Exact class, duplicate-call, duplicate-row, and multiplicity
    ///         counts for the current used call prefix.
    /// @invariant The schedule and every proof-flow input remain unchanged.
    Phase2FilterClassCensus
    Phase2FilterScheduleArena::measureClassReuse() const {
        Phase2FilterClassCensus census{};
        census.callCount = static_cast<uint32_t>(calls.size());
        census.uniqueClassCount = static_cast<uint32_t>(classes.size());
        census.uniqueExaminedRows = classExaminedRows;
        for (uint32_t multiplicity : classMultiplicities) {
            census.maximumClassMultiplicity = std::max(
                census.maximumClassMultiplicity, multiplicity);
        }
        assert(census.uniqueClassCount <= census.callCount);
        assert(census.uniqueExaminedRows <= examinedRows);
        census.duplicateCallCount =
            census.callCount - census.uniqueClassCount;
        census.duplicateExaminedRows =
            examinedRows - census.uniqueExaminedRows;
        return census;
    }

    /// @brief Reserve the request-growth call schedule once.
    ///
    /// @details
    /// Requires a positive measured ceiling, reserves exactly that many
    /// pointer-free descriptors, and leaves the used prefix empty.
    ///
    /// @param fixedCapacity Immutable request-call ceiling.
    /// @return An empty reusable growth schedule.
    /// @invariant No later method changes the vector allocation capacity.
    Phase2GrowthScheduleArena::Phase2GrowthScheduleArena(
        uint32_t fixedCapacity)
        : capacity(fixedCapacity) {
        assert(capacity > 0);
        calls.reserve(capacity);
    }

    /// @brief Clear the request-growth schedule without releasing storage.
    ///
    /// @details
    /// Resets only the used prefix. The constructor reservation remains
    /// available for the next Phase 2 sweep.
    ///
    /// @return Nothing.
    /// @invariant The vector capacity remains at least `capacity`.
    void Phase2GrowthScheduleArena::clear() {
        calls.clear();
        assert(calls.capacity() >= capacity);
    }

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
    DevicePhase2GrowthCall Phase2GrowthScheduleArena::appendCall(
        uint32_t taskIndex,
        uint32_t batchIndex,
        uint32_t filterCallIndex) {
        assert(calls.size() < capacity);
        const DevicePhase2GrowthCall call{
            taskIndex, batchIndex, filterCallIndex };
        calls.push_back(call);
        return call;
    }

}  // namespace gl::gpu
