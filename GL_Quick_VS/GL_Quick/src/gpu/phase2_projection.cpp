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
#include "../compiler/compiled_definition_registry.hpp"
#include "../compiler.hpp"
#include "../parameters.hpp"
#include "../infra/hashburst_dump.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <thread>
#include <type_traits>

namespace gl::gpu {

    /// @brief Temporary byte-key dictionary for one logical-block projection.
    /// @details Owns only construction scratch. Map membership and value runs
    /// remain in their separate projected tables; equal bytes share one offset.
    struct Phase2ProjectionKeyStore {
        LbArena arena{ &gpuHostStorage() };
        DirtyState dirty = DirtyState::Clean;
        ColdHashMap<BytesKeyStore, uint32_t> offsets{ &arena, &dirty };
    };


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
        capacity.byteMapViews += 4 * capacity.logicalBlocks;
        assert(capacity.logicalBlocks == kMaxProjectedBlocksPerChunk);
        assert(capacity.statements > 0);
        return capacity;
    }

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
        uint32_t* ends, uint32_t endCapacity) {
        assert(endCapacity >= count);
        assert(count == 0 || (usages != nullptr && ends != nullptr));
        constexpr uint64_t Phase2ProjectionUsage::* usageFields[] = {
            &Phase2ProjectionUsage::logicalBlocks,
            &Phase2ProjectionUsage::statements,
            &Phase2ProjectionUsage::nameRecords,
            &Phase2ProjectionUsage::nameBytes,
            &Phase2ProjectionUsage::nameSlots,
            &Phase2ProjectionUsage::ruleStringRecords,
            &Phase2ProjectionUsage::ruleStringBytes,
            &Phase2ProjectionUsage::byteMapViews,
            &Phase2ProjectionUsage::byteMapEntries,
            &Phase2ProjectionUsage::byteMapSlots,
            &Phase2ProjectionUsage::byteKeyBytes,
            &Phase2ProjectionUsage::blobRecords,
            &Phase2ProjectionUsage::blobBytes,
            &Phase2ProjectionUsage::reverseMapViews,
            &Phase2ProjectionUsage::reverseMapEntries,
            &Phase2ProjectionUsage::reverseMapSlots,
            &Phase2ProjectionUsage::reverseKeyBytes,
            &Phase2ProjectionUsage::reverseOwners,
            &Phase2ProjectionUsage::podMapViews,
            &Phase2ProjectionUsage::podMapEntries,
            &Phase2ProjectionUsage::podMapSlots,
            &Phase2ProjectionUsage::podRunValues,
            &Phase2ProjectionUsage::mandatoryStatementKeys,
            &Phase2ProjectionUsage::metadataBytes
        };
        constexpr uint32_t Phase2ProjectionCapacity::* capacityFields[] = {
            &Phase2ProjectionCapacity::logicalBlocks,
            &Phase2ProjectionCapacity::statements,
            &Phase2ProjectionCapacity::nameRecords,
            &Phase2ProjectionCapacity::nameBytes,
            &Phase2ProjectionCapacity::nameSlots,
            &Phase2ProjectionCapacity::ruleStringRecords,
            &Phase2ProjectionCapacity::ruleStringBytes,
            &Phase2ProjectionCapacity::byteMapViews,
            &Phase2ProjectionCapacity::byteMapEntries,
            &Phase2ProjectionCapacity::byteMapSlots,
            &Phase2ProjectionCapacity::byteKeyBytes,
            &Phase2ProjectionCapacity::blobRecords,
            &Phase2ProjectionCapacity::blobBytes,
            &Phase2ProjectionCapacity::reverseMapViews,
            &Phase2ProjectionCapacity::reverseMapEntries,
            &Phase2ProjectionCapacity::reverseMapSlots,
            &Phase2ProjectionCapacity::reverseKeyBytes,
            &Phase2ProjectionCapacity::reverseOwners,
            &Phase2ProjectionCapacity::podMapViews,
            &Phase2ProjectionCapacity::podMapEntries,
            &Phase2ProjectionCapacity::podMapSlots,
            &Phase2ProjectionCapacity::podRunValues,
            &Phase2ProjectionCapacity::mandatoryStatementKeys,
            &Phase2ProjectionCapacity::metadataBytes
        };
        static_assert(sizeof(Phase2ProjectionUsage) == 24 * sizeof(uint64_t));
        static_assert(sizeof(Phase2ProjectionCapacity) == 24 * sizeof(uint32_t));
        uint64_t totals[24]{};
        uint32_t waves = 0;
        for (uint32_t i = 0; i < count; ++i) {
            assert(usages[i].logicalBlocks == 1);
            bool fits = true;
            for (uint32_t field = 0; field < 24; ++field) {
                const uint64_t add = usages[i].*usageFields[field];
                const uint64_t ceiling = capacity.*capacityFields[field];
                assert(add <= ceiling && "one logical block exceeds its construction shard capacity");
                if (totals[field] + add > ceiling) fits = false;
            }
            if (!fits) {
                assert(i != 0 && (waves == 0 || ends[waves - 1] < i));
                ends[waves++] = i;
                for (uint64_t& total : totals) total = 0;
            }
            for (uint32_t field = 0; field < 24; ++field)
                totals[field] += usages[i].*usageFields[field];
        }
        if (count != 0) ends[waves++] = count;
        return waves;
    }

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
        uint32_t selectedHashMemories, bool buildPrefixFilter) {
        assert(selectedHashMemories != 0 && (selectedHashMemories & ~0x0fu) == 0);
        LbArena arena(&gpuHostStorage());
        arena.setPurpose(PoolPurpose::GpuHost);
        DirtyState dirty = DirtyState::Clean;
        ColdHashSet<BytesKeyStore> keys(&arena, &dirty);
        const HashMemory* const memories[4] = { &body.overallHashMemory,
            &body.localHashMemory, &body.localHashMemoryDelta, &body.workingMemory };
        const auto visitMap = [](const auto& map, const auto& visit) {
            for (int32_t id = 1; id <= map.count(); ++id) visit(map.keyAt(id));
        };
        const auto visitSelected = [&](uint32_t memory, const auto& visit) {
            if ((selectedHashMemories & (1u << memory)) != 0) {
                visitMap(memories[memory]->normalizedEncodedKeys, visit);
                visitMap(memories[memory]->normalizedEncodedSubkeys, visit);
            }
        };
        const auto remember = [&](StrSpan key) { keys.mint(key); };
        for (uint32_t memory = 0; memory < 4; ++memory) visitSelected(memory, remember);
        visitMap(body.overallHashMemory.encodedMap, remember);
        visitMap(body.overallHashMemory.remainingArgsNormalizedEncodedMap, remember);
        Phase2ProjectionUsage usage{};
        if (buildPrefixFilter) {
            PagedVector<NameId> arities(&arena, &dirty);
            const int32_t names = body.nameMap.nameCount();
            for (int32_t id = 0; id <= names; ++id) arities.push_back(-1);
            const auto wordAt = [](StrSpan key, int32_t word) {
                assert(word >= 0 && (word + 1) * sizeof(NameId)
                    <= static_cast<std::size_t>(key.len));
                NameId value = 0;
                std::memcpy(&value, key.ptr + word * sizeof(NameId), sizeof(NameId));
                return value;
            };
            const auto recordArity = [&](StrSpan key) {
                assert(key.len >= 4 * sizeof(NameId));
                if (wordAt(key, 0) != 1) return;
                const NameId length = wordAt(key, 1);
                assert(length >= 2 && length % 2 == 0);
                assert(key.len == (length + 2) * sizeof(NameId));
                const NameId core = wordAt(key, 2);
                assert(core > 0 && core <= names);
                const NameId arity = (length - 2) / 2;
                assert(arity <= ExecutionParameters::MAX_ARITY);
                assert(arities[core] == -1 || arities[core] == arity);
                arities.setAt(core, arity);
            };
            for (uint32_t memory = 0; memory < 4; ++memory)
                visitSelected(memory, recordArity);
            for (uint32_t memory = 0; memory < 4; ++memory) {
                LbArena prefixArena(&gpuHostStorage());
                prefixArena.setPurpose(PoolPurpose::GpuHost);
                DirtyState prefixDirty = DirtyState::Clean;
                ColdHashSet<BytesKeyStore> prefixes(&prefixArena, &prefixDirty);
                ColdHashSet<PodKeyStore<uint64_t>> transitions(&prefixArena, &prefixDirty);
                const auto recordPrefix = [&](StrSpan key) {
                    const NameId count = wordAt(key, 0);
                    const NameId length = wordAt(key, 1);
                    assert(count > 0 && count <= ExecutionParameters::MAX_EXPRESSIONS);
                    assert(length >= 2 * count && length <= ExecutionParameters::MAX_KEY_SLOTS);
                    assert(key.len == (length + 2) * sizeof(NameId));
                    NameId prefixLength = 0;
                    for (NameId premise = 0; premise < count - 1; ++premise) {
                        assert(prefixLength + 2 <= length);
                        const NameId core = wordAt(key, 2 + prefixLength);
                        assert(core > 0 && core <= names && arities[core] >= 0);
                        prefixLength += 2 + 2 * arities[core];
                    }
                    assert(prefixLength + 2 <= length);
                    const NameId core = wordAt(key, 2 + prefixLength);
                    const NameId polarity = wordAt(key, 3 + prefixLength);
                    assert(core > 0 && core <= names && arities[core] >= 0);
                    assert(polarity == 0 || polarity == 1);
                    assert(prefixLength + 2 + 2 * arities[core] == length);
                    NameId prefix[2 + ExecutionParameters::MAX_KEY_SLOTS];
                    prefix[0] = count - 1;
                    prefix[1] = prefixLength;
                    std::memcpy(prefix + 2, key.ptr + 2 * sizeof(NameId),
                        prefixLength * sizeof(NameId));
                    const StrSpan bytes(reinterpret_cast<const char*>(prefix),
                        static_cast<int32_t>((prefixLength + 2) * sizeof(NameId)));
                    const int32_t id = prefixes.mint(bytes);
                    keys.mint(bytes);
                    transitions.mint((static_cast<uint64_t>(id) << 32)
                        | static_cast<uint32_t>(core));
                };
                visitSelected(memory, recordPrefix);
                const uint64_t count = static_cast<uint64_t>(prefixes.count());
                uint64_t slots = count == 0 ? 0 : 2;
                while (slots < count * 2) slots <<= 1;
                usage.byteMapEntries += count;
                usage.byteMapSlots += slots;
                usage.blobRecords += count;
                usage.blobBytes += static_cast<uint64_t>(transitions.count()) * 8;
            }
        }
        usage.byteKeyBytes = static_cast<uint64_t>(keys.keyStore().logicalByteCount());
        return usage;
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
    /// metadata. It uses private arena scratch without source mutation, proof decisions, or lookup
    /// substitution; callers may reduce the returned integers in any order.
    ///
    /// @param body Resident logical block whose post-Phase-1 read image is measured.
    /// @param anchorInfo Immutable anchor information used to measure its name bytes.
    /// @param selectedHashMemories Bit `1 << DeviceHashMemoryKind` for every
    ///                 request memory consumed by this logical block's tasks.
    /// @param buildPrefixFilter Include bounded derived next-predicate maps.
    /// @return Exact direct-column usage plus safe derived-index upper bounds.
    /// @invariant The result is a pure function of the resident read image. It
    ///            may bound deterministic transfer chunks but never changes a
    ///            proof gate, request, or verdict.
    Phase2ProjectionUsage measurePhase2ProjectionUsage(
        const Memory& body,
        const ce::AnchorInfo& anchorInfo,
        uint32_t selectedHashMemories,
        bool buildPrefixFilter) {
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
            if (buildPrefixFilter) ++usage.byteMapViews;
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

        const Phase2ProjectionUsage sharedKeys = measurePhase2SharedKeyUsage(
            body, selectedHashMemories, buildPrefixFilter);
        usage.byteKeyBytes = sharedKeys.byteKeyBytes;
        usage.byteMapEntries += sharedKeys.byteMapEntries;
        usage.byteMapSlots += sharedKeys.byteMapSlots;
        usage.blobRecords += sharedKeys.blobRecords;
        usage.blobBytes += sharedKeys.blobBytes;

        usage.reverseMapViews = 1;
        const int32_t reverseEdgeCount = remainingArgs.inner().blobCount();
        const int32_t reverseEdgeBytes = remainingArgs.inner().poolByteCount();
        assert(reverseEdgeCount >= 0 && reverseEdgeBytes >= 0);
        usage.reverseMapEntries += static_cast<uint64_t>(reverseEdgeCount);
        usage.reverseOwners += static_cast<uint64_t>(reverseEdgeCount);
        usage.reverseKeyBytes = 0; // Reverse keys reference their forward blobs.
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

        usage.mandatoryStatementKeys = static_cast<uint64_t>(
            body.intLocalEncodedStatements.size())
            + static_cast<uint64_t>(body.intLocalEncodedStatementsDelta.size())
            + static_cast<uint64_t>(body.intExternalStatements.size());
        const StrSpan expressionKey = body.exprKeyView();
        assert(expressionKey.len >= 0);
        usage.metadataBytes = static_cast<uint64_t>(expressionKey.len)
            + static_cast<uint64_t>(anchorInfo.name.size());
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
            usage.tasks = 1;
            usage.batches = 1;
            return phase2TaskUsageWithStumps(usage, stumpCount, true);
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
        if (usage.batches != 0) usage.tasks = 1;
        return phase2TaskUsageWithStumps(usage, stumpCount, false);
    }

    /// @brief One executor part's task usage from its logical block's
    ///        stump-free usage and the part's stump count.
    ///
    /// @param stumpFree The block's usage measured with a stump count of zero.
    /// @param stumpCount Number of stumps owned by this executor part.
    /// @param counterExampleMode Whether this is the unsplit counter-example route.
    /// @return The part's task usage.
    Phase2TaskProjectionUsage phase2TaskUsageWithStumps(
        const Phase2TaskProjectionUsage& stumpFree,
        uint32_t stumpCount,
        bool counterExampleMode) {
        assert(stumpFree.stumps == 0);
        Phase2TaskProjectionUsage usage = stumpFree;
        if (counterExampleMode) {
            assert(stumpCount == 0);
            return usage;
        }
        if (usage.tasks != 0) usage.stumps = stumpCount;
        return usage;
    }

    /// @brief The hash memories one logical block's executable request batches
    ///        read.
    ///
    /// @param body Resident logical block.
    /// @param counterExampleMode Whether this is the unsplit counter-example route.
    /// @return The selection mask; zero when the block has no executable batch.
    uint32_t selectPhase2HashMemories(const Memory& body, bool counterExampleMode) {
        const auto bit = [](DeviceHashMemoryKind memory) {
            return 1u << static_cast<uint32_t>(memory);
        };
        if (counterExampleMode) return bit(DeviceHashMemoryKind::overall);
        uint32_t selected = 0;
        if (!body.workingMemory.encodedMap.empty()
            && !body.intLocalEncodedStatements.empty())
            selected |= bit(DeviceHashMemoryKind::working);
        if (!body.intLocalEncodedStatementsDelta.empty()
            || (!body.intExternalStatements.empty()
                && !body.intLocalEncodedStatements.empty()))
            selected |= bit(DeviceHashMemoryKind::overall);
        if (!body.localHashMemory.encodedMap.empty()
            && !body.intExternalStatements.empty())
            selected |= bit(DeviceHashMemoryKind::local);
        if (!body.localHashMemoryDelta.encodedMap.empty())
            selected |= bit(DeviceHashMemoryKind::localDelta);
        return selected;
    }

    /// @brief Field-wise sum of two projection capacities.
    ///
    /// @details
    /// Lays construction-shard capacities end to end
    /// (D-425). A sum that leaves `uint32_t` asserts.
    ///
    /// @param left First capacity.
    /// @param right Second capacity.
    /// @return The field-wise sum.
    Phase2ProjectionCapacity addProjectionCapacity(const Phase2ProjectionCapacity& left,
        const Phase2ProjectionCapacity& right) {
        const auto add = [](uint32_t a, uint32_t b) -> uint32_t {
            assert(b <= std::numeric_limits<uint32_t>::max() - a);
            return a + b;
        };
        Phase2ProjectionCapacity sum{};
        sum.logicalBlocks = add(left.logicalBlocks, right.logicalBlocks);
        sum.statements = add(left.statements, right.statements);
        sum.nameRecords = add(left.nameRecords, right.nameRecords);
        sum.nameBytes = add(left.nameBytes, right.nameBytes);
        sum.nameSlots = add(left.nameSlots, right.nameSlots);
        sum.ruleStringRecords = add(left.ruleStringRecords, right.ruleStringRecords);
        sum.ruleStringBytes = add(left.ruleStringBytes, right.ruleStringBytes);
        sum.byteMapViews = add(left.byteMapViews, right.byteMapViews);
        sum.byteMapEntries = add(left.byteMapEntries, right.byteMapEntries);
        sum.byteMapSlots = add(left.byteMapSlots, right.byteMapSlots);
        sum.byteKeyBytes = add(left.byteKeyBytes, right.byteKeyBytes);
        sum.blobRecords = add(left.blobRecords, right.blobRecords);
        sum.blobBytes = add(left.blobBytes, right.blobBytes);
        sum.reverseMapViews = add(left.reverseMapViews, right.reverseMapViews);
        sum.reverseMapEntries = add(left.reverseMapEntries, right.reverseMapEntries);
        sum.reverseMapSlots = add(left.reverseMapSlots, right.reverseMapSlots);
        sum.reverseKeyBytes = add(left.reverseKeyBytes, right.reverseKeyBytes);
        sum.reverseOwners = add(left.reverseOwners, right.reverseOwners);
        sum.podMapViews = add(left.podMapViews, right.podMapViews);
        sum.podMapEntries = add(left.podMapEntries, right.podMapEntries);
        sum.podMapSlots = add(left.podMapSlots, right.podMapSlots);
        sum.podRunValues = add(left.podRunValues, right.podRunValues);
        sum.mandatoryStatementKeys = add(left.mandatoryStatementKeys, right.mandatoryStatementKeys);
        sum.metadataBytes = add(left.metadataBytes, right.metadataBytes);
        return sum;
    }

    /// @brief Assert that every field of `capacity` is positive.
    ///
    /// @param capacity Ceilings or storage sizes to check.
    /// @return Nothing.
    void Phase2ProjectionArena::assertPositiveCapacity(
        const Phase2ProjectionCapacity& capacity) {
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
    }

    /// @brief Measure exact aligned storage for one projection role.
    /// @details Uses the same ordered field census as reserveStorage.
    /// @param fixedCapacity Logical column and scratch ceilings.
    /// @param storage Physical backing-column ceilings.
    /// @param role Owning, backing, slice or retained sealing storage.
    /// @return Measured payload layout, without block rounding.
    GpuHostLayout Phase2ProjectionArena::hostLayout(
        const Phase2ProjectionCapacity& fixedCapacity,
        const Phase2ProjectionCapacity& storage, Role role) {
        GpuHostLayout layout;
        if (role == Role::Sealing) {
            layout.add<DeviceLogicalBlockProjection>(fixedCapacity.logicalBlocks);
            layout.add<NameId>(fixedCapacity.statements);
            layout.add<DeviceNameRecord>(fixedCapacity.nameRecords);
            layout.add<char>(fixedCapacity.nameBytes);
            layout.add<DeviceRuleStringRecord>(fixedCapacity.ruleStringRecords);
            layout.add<char>(fixedCapacity.ruleStringBytes);
        }
        if (role == Role::Owning || role == Role::Backing) {
            layout.add<DeviceLogicalBlockProjection>(storage.logicalBlocks);
            layout.add<IntEncodedExpr>(storage.statements);
            layout.add<NameId>(storage.statements);
            layout.add<DeviceNameRecord>(storage.nameRecords);
            layout.add<char>(storage.nameBytes);
            layout.add<int32_t>(storage.nameSlots);
            layout.add<DeviceRuleStringRecord>(storage.ruleStringRecords);
            layout.add<char>(storage.ruleStringBytes);
            layout.add<DeviceByteMapView>(storage.byteMapViews);
            layout.add<DeviceByteMapEntry>(storage.byteMapEntries);
            layout.add<int32_t>(storage.byteMapSlots);
            layout.add<char>(storage.byteKeyBytes);
            layout.add<DeviceBlobRecord>(storage.blobRecords);
            layout.add<char>(storage.blobBytes);
            layout.add<DeviceReverseMapView>(storage.reverseMapViews);
            layout.add<DeviceReverseMapEntry>(storage.reverseMapEntries);
            layout.add<int32_t>(storage.reverseMapSlots);
            layout.add<char>(storage.reverseKeyBytes);
            layout.add<int32_t>(storage.reverseOwners);
            layout.add<DevicePodMapView>(storage.podMapViews);
            layout.add<DevicePodMapEntry>(storage.podMapEntries);
            layout.add<int32_t>(storage.podMapSlots);
            layout.add<int32_t>(storage.podRunValues);
            layout.add<int64_t>(storage.mandatoryStatementKeys);
            layout.add<char>(storage.metadataBytes);
        }
        if (role == Role::Owning || role == Role::Slice) {
            layout.add<NameId>(fixedCapacity.nameRecords);
            layout.add<uint8_t>(fixedCapacity.nameRecords);
            layout.add<NameId>(fixedCapacity.nameRecords);
            layout.add<DecodedRankRange>(fixedCapacity.nameRecords);
            layout.add<uint32_t>(fixedCapacity.reverseMapEntries);
            layout.add<SharedNameProjection>(fixedCapacity.logicalBlocks);
            layout.add<SharedRuleStringProjection>(fixedCapacity.logicalBlocks);
        }
        return layout;
    }

    /// @brief Reserve all owned columns in one measured contiguous lease.
    /// @details Slice columns retain their backing views; only scratch is owned.
    /// @param storage Physical backing-column capacities.
    /// @return Nothing.
    void Phase2ProjectionArena::reserveStorage(const Phase2ProjectionCapacity& storage) {
        const auto layout = hostLayout(capacity, storage, role_);
        hostStorage_.reserve(layout.bytes);
        const auto reserve = [&](auto& column, uint32_t count) {
            column.bindStorage(hostStorage_);
            column.reserve(count);
        };
        if (role_ == Role::Sealing) {
            reserve(logicalBlocks, capacity.logicalBlocks);
            reserve(sourceStatementIndices, capacity.statements);
            reserve(nameRecords, capacity.nameRecords);
            reserve(nameBytes, capacity.nameBytes);
            reserve(ruleStringRecords, capacity.ruleStringRecords);
            reserve(ruleStringBytes, capacity.ruleStringBytes);
        }
        if (role_ == Role::Owning || role_ == Role::Backing) {
            reserve(logicalBlocks, storage.logicalBlocks);
            reserve(statements, storage.statements);
            reserve(sourceStatementIndices, storage.statements);
            reserve(nameRecords, storage.nameRecords);
            reserve(nameBytes, storage.nameBytes);
            reserve(nameSlots, storage.nameSlots);
            reserve(ruleStringRecords, storage.ruleStringRecords);
            reserve(ruleStringBytes, storage.ruleStringBytes);
            reserve(byteMapViews, storage.byteMapViews);
            reserve(byteMapEntries, storage.byteMapEntries);
            reserve(byteMapSlots, storage.byteMapSlots);
            reserve(byteKeyBytes, storage.byteKeyBytes);
            reserve(blobRecords, storage.blobRecords);
            reserve(blobBytes, storage.blobBytes);
            reserve(reverseMapViews, storage.reverseMapViews);
            reserve(reverseMapEntries, storage.reverseMapEntries);
            reserve(reverseMapSlots, storage.reverseMapSlots);
            reserve(reverseKeyBytes, storage.reverseKeyBytes);
            reserve(reverseOwners, storage.reverseOwners);
            reserve(podMapViews, storage.podMapViews);
            reserve(podMapEntries, storage.podMapEntries);
            reserve(podMapSlots, storage.podMapSlots);
            reserve(podRunValues, storage.podRunValues);
            reserve(mandatoryStatementKeys, storage.mandatoryStatementKeys);
            reserve(metadataBytes, storage.metadataBytes);
        }
        if (role_ == Role::Owning || role_ == Role::Slice) {
            reserve(decodedRankScratch_, capacity.nameRecords);
            reserve(usedAsPremiseScratch_, capacity.nameRecords);
            reserve(decodedRankWorkScratch_, capacity.nameRecords);
            reserve(decodedRankRangeScratch_, capacity.nameRecords);
            reserve(reverseRunScratch_, capacity.reverseMapEntries);
            reserve(sharedNameProjections_, capacity.logicalBlocks);
            reserve(sharedRuleStringProjections_, capacity.logicalBlocks);
        }
    }


    /// @brief Reserve every projection array to an immutable capacity ceiling.
    ///
    /// @details
    /// Asserts positive capacities, reserves all semantic columns plus the
    /// construction scratch once, and leaves every used length zero. Later
    /// appends assert before crossing any ceiling.
    ///
    /// @param fixedCapacity Element and byte ceilings for this arena lifetime.
    /// @return A reusable empty arena with all storage reserved.
    /// @invariant Column capacities are at least their corresponding declared
    ///            ceilings for the arena's entire lifetime.
    Phase2ProjectionArena::Phase2ProjectionArena(
        Phase2ProjectionCapacity fixedCapacity)
        : capacity(fixedCapacity) {
        assertPositiveCapacity(capacity);
        reserveStorage(capacity);
    }

    /// @brief Reserve the columns of a projection whose construction shards
    ///        write straight into it.
    ///
    /// @details
    /// Every column reserves `storage` elements while `capacity` stays the
    /// ceiling the adopted totals must fit. No construction scratch: a backing
    /// arena never appends logical blocks itself (D-425).
    ///
    /// @param fixedCapacity Ceilings of the adopted totals; every field positive.
    /// @param storage Elements each column reserves; every field positive.
    /// @return An empty backing arena.
    Phase2ProjectionArena::Phase2ProjectionArena(
        Phase2ProjectionCapacity fixedCapacity, const Phase2ProjectionCapacity& storage)
        : capacity(fixedCapacity), role_(Role::Backing) {
        assertPositiveCapacity(capacity);
        assertPositiveCapacity(storage);
        reserveStorage(storage);
    }

    /// @brief Reserve the host lookup image retained across streamed uploads.
    ///
    /// @details Only sealing inputs (including the rule copy per blob record)
    /// and source ordinals have physical backing; separate bounded construction
    /// windows produce the device-only columns.
    /// @param fixedCapacity Complete chunk limits for retained columns.
    /// @param mode Explicit retained-only constructor selection.
    /// @return An empty retained image with no construction scratch.
    Phase2ProjectionArena::Phase2ProjectionArena(
        Phase2ProjectionCapacity fixedCapacity, SealingOnly mode)
        : capacity(fixedCapacity), role_(Role::Sealing) {
        (void)mode;
        assertPositiveCapacity(capacity);
        reserveStorage(capacity);
    }

    /// @brief Build a construction shard as a slice of `backing`'s columns.
    ///
    /// @details
    /// Every column views `sliceCapacity` elements of the matching `backing`
    /// column from `sliceStart` on; construction scratch is reserved as for an
    /// owning arena (D-425).
    ///
    /// @param sliceCapacity Ceilings of this shard; every field positive.
    /// @param backing The backing arena whose storage this shard views.
    /// @param sliceStart First element of this shard's slice in each column.
    /// @return An empty slice arena.
    Phase2ProjectionArena::Phase2ProjectionArena(Phase2ProjectionCapacity sliceCapacity,
        Phase2ProjectionArena& backing, const Phase2ProjectionCapacity& sliceStart)
        : capacity(sliceCapacity), role_(Role::Slice) {
        assertPositiveCapacity(capacity);
        assert(backing.role_ == Role::Backing && !backing.adopted_);
        logicalBlocks.attachSlice(backing.logicalBlocks, sliceStart.logicalBlocks, capacity.logicalBlocks);
        statements.attachSlice(backing.statements, sliceStart.statements, capacity.statements);
        sourceStatementIndices.attachSlice(backing.sourceStatementIndices, sliceStart.statements, capacity.statements);
        nameRecords.attachSlice(backing.nameRecords, sliceStart.nameRecords, capacity.nameRecords);
        nameBytes.attachSlice(backing.nameBytes, sliceStart.nameBytes, capacity.nameBytes);
        nameSlots.attachSlice(backing.nameSlots, sliceStart.nameSlots, capacity.nameSlots);
        ruleStringRecords.attachSlice(backing.ruleStringRecords, sliceStart.ruleStringRecords, capacity.ruleStringRecords);
        ruleStringBytes.attachSlice(backing.ruleStringBytes, sliceStart.ruleStringBytes, capacity.ruleStringBytes);
        byteMapViews.attachSlice(backing.byteMapViews, sliceStart.byteMapViews, capacity.byteMapViews);
        byteMapEntries.attachSlice(backing.byteMapEntries, sliceStart.byteMapEntries, capacity.byteMapEntries);
        byteMapSlots.attachSlice(backing.byteMapSlots, sliceStart.byteMapSlots, capacity.byteMapSlots);
        byteKeyBytes.attachSlice(backing.byteKeyBytes, sliceStart.byteKeyBytes, capacity.byteKeyBytes);
        blobRecords.attachSlice(backing.blobRecords, sliceStart.blobRecords, capacity.blobRecords);
        blobBytes.attachSlice(backing.blobBytes, sliceStart.blobBytes, capacity.blobBytes);
        reverseMapViews.attachSlice(backing.reverseMapViews, sliceStart.reverseMapViews, capacity.reverseMapViews);
        reverseMapEntries.attachSlice(backing.reverseMapEntries, sliceStart.reverseMapEntries, capacity.reverseMapEntries);
        reverseMapSlots.attachSlice(backing.reverseMapSlots, sliceStart.reverseMapSlots, capacity.reverseMapSlots);
        reverseKeyBytes.attachSlice(backing.reverseKeyBytes, sliceStart.reverseKeyBytes, capacity.reverseKeyBytes);
        reverseOwners.attachSlice(backing.reverseOwners, sliceStart.reverseOwners, capacity.reverseOwners);
        podMapViews.attachSlice(backing.podMapViews, sliceStart.podMapViews, capacity.podMapViews);
        podMapEntries.attachSlice(backing.podMapEntries, sliceStart.podMapEntries, capacity.podMapEntries);
        podMapSlots.attachSlice(backing.podMapSlots, sliceStart.podMapSlots, capacity.podMapSlots);
        podRunValues.attachSlice(backing.podRunValues, sliceStart.podRunValues, capacity.podRunValues);
        mandatoryStatementKeys.attachSlice(backing.mandatoryStatementKeys, sliceStart.mandatoryStatementKeys, capacity.mandatoryStatementKeys);
        metadataBytes.attachSlice(backing.metadataBytes, sliceStart.metadataBytes, capacity.metadataBytes);
        reserveStorage(capacity);
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
        sourceStatementIndices.clear();
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
        if (role_ == Role::Sealing) {
            assert(logicalBlocks.capacity() >= capacity.logicalBlocks);
            assert(sourceStatementIndices.capacity() >= capacity.statements);
            assert(nameRecords.capacity() >= capacity.nameRecords);
            assert(nameBytes.capacity() >= capacity.nameBytes);
            assert(ruleStringRecords.capacity() >= capacity.ruleStringRecords);
            assert(ruleStringBytes.capacity() >= capacity.ruleStringBytes);
        } else {
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
    /// @param definitions Immutable compiled-core registry.
    /// @param anchorInfo Immutable anchor information.
    /// @param parameters Immutable phase-two evaluation parameters.
    /// @param selectedHashMemories Bit `1 << DeviceHashMemoryKind` for every
    ///                 request memory consumed by this logical block's tasks.
    /// @param buildPrefixFilter Build exact-prefix next-predicate maps for growth.
    /// @return Pointer-free descriptor also appended to `logicalBlocks`.
    /// @invariant The appended slices decode byte-identically to `body` and
    ///            preserve statement index order and local identifier values.
    DeviceLogicalBlockProjection Phase2ProjectionArena::appendLogicalBlock(
        const Memory& body,
        const CompiledDefinitionRegistry& definitions,
            const ce::AnchorInfo& anchorInfo,
            const ProverParameters& parameters,
        uint32_t selectedHashMemories,
        bool buildPrefixFilter) {
        const auto preflightStarted = std::chrono::steady_clock::now();
        assert(role_ == Role::Owning || role_ == Role::Slice);
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
        const auto compiledCategory = [&definitions](
            const StrSpan& value) -> DeviceCompiledCategory {
            const StrSpan core = extractExpressionSpan(value);
            if (core.len == 0) return DeviceCompiledCategory::absent;
            const LogicalEntity* entity = definitions.compiledEntity(core);
            if (entity == nullptr) return DeviceCompiledCategory::absent;
            return equalSpans(StrSpan(entity->category), StrSpan("atomic", 6))
                ? DeviceCompiledCategory::atomic
                : DeviceCompiledCategory::nonAtomic;
        };

        const int32_t statementCountSigned = body.intEncodedStatementsCount();
        assert(statementCountSigned >= 0);
        const std::size_t coreSlots =
            static_cast<std::size_t>(body.nameMap.nameCount()) + 1;
        assert(coreSlots <= capacity.nameRecords);
        usedAsPremiseScratch_.assign(coreSlots, 0);
        forEachPremiseCore(body, [&](NameId core, int32_t polarity) {
            usedAsPremiseScratch_[core] |= static_cast<uint8_t>(1u << polarity);
        });
        // A row is uploaded when its signed core is a premise core and some
        // live rule premise of the block accepts it on its u_ literals,
        // repetitions and scope (premisePatternAdmits) — the processor
        // request filter applies the same union, so both routes keep one row
        // list and split stumps translate between them.
        const auto usedAsPremise = [&](const IntEncodedExpr& row) {
            assert(row.nameId > 0 && static_cast<std::size_t>(row.nameId) < coreSlots);
            assert(row.negation == 0 || row.negation == 1);
            return (usedAsPremiseScratch_[row.nameId] & (1u << row.negation)) != 0
                && premisePatternAdmits(body, row);
        };
        std::size_t statementCount = 0;
        for (int32_t index = 0; index < statementCountSigned; ++index)
            if (usedAsPremise(body.intEncodedStatements[index])) ++statementCount;
        assert(statements.size() + statementCount <= capacity.statements);

        const int32_t nameCountSigned = body.nameMap.nameCount();
        assert(nameCountSigned >= 0);
        const std::size_t localNameRecordCount =
            static_cast<std::size_t>(nameCountSigned) + 1;
        const int64_t bodyNameBytes =
            body.nameMap.names->keyStore().logicalByteCount();
        assert(bodyNameBytes >= 0);
        // Sharing compares against this arena's own projected copy of each
        // earlier block's names, never the earlier block itself: that block
        // was released after its projection and may be deloaded by now.
        const SharedNameProjection* sharedName = nullptr;
        for (const SharedNameProjection& candidate
             : sharedNameProjections_) {
            if (candidate.recordCount != localNameRecordCount) continue;
            if (static_cast<int64_t>(candidate.byteCount) != bodyNameBytes)
                continue;
            bool equal = true;
            for (NameId id = 1; id <= nameCountSigned; ++id) {
                const DeviceNameRecord& record = nameRecords[
                    candidate.recordOffset + static_cast<uint32_t>(id)];
                if (record.parentId != body.nameMap.parentOf(id)
                    || !equalSpans(StrSpan(nameBytes.data() + record.byteOffset,
                                       static_cast<int32_t>(record.byteLength)),
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
            if (usedAsPremise(body.intEncodedStatements[index])) {
                statements.push_back(body.intEncodedStatements[index]);
                sourceStatementIndices.push_back(index);
            }

        // Validate the complete stable filtered slice, including its exact
        // cardinality; excluded source rows do not consume device indices.
        std::size_t retained = descriptor.statementOffset;
        for (int32_t index = 0; index < statementCountSigned; ++index) {
            const IntEncodedExpr& source = body.intEncodedStatements[index];
            if (!usedAsPremise(source)) continue;
            assert(retained < statements.size());
            assert(std::memcmp(&statements[retained], &source, sizeof(source)) == 0);
            ++retained;
        }
        assert(retained == statements.size());
        assert(retained - descriptor.statementOffset == descriptor.statementCount);
        assert(sourceStatementIndices.size() == statements.size());
        // Rule-14 dump of the target block: the rows the device receives and
        // the premise-core rows the premise patterns held back.
        if (hashburst_dump::isTargetLB(body)) {
            hashburst_dump::recordPhase2Upload(body,
                sourceStatementIndices.data() + descriptor.statementOffset,
                descriptor.statementCount, usedAsPremiseScratch_.data(), coreSlots);
        }

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
        // Compared against this arena's projected copy, like the names.
        const SharedRuleStringProjection* sharedRuleStrings = nullptr;
        for (const SharedRuleStringProjection& candidate
             : sharedRuleStringProjections_) {
            if (candidate.recordCount != localRuleStringRecordCount) continue;
            if (static_cast<int64_t>(candidate.byteCount)
                != bodyRuleStringBytes) continue;
            bool equal = true;
            for (int32_t id = 1; id <= ruleStringCountSigned; ++id) {
                const DeviceRuleStringRecord& record = ruleStringRecords[
                    candidate.recordOffset + static_cast<uint32_t>(id)];
                if (!equalSpans(StrSpan(ruleStringBytes.data() + record.byteOffset,
                                    static_cast<int32_t>(record.byteLength)),
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
            descriptor.ruleStringRecordOffset,
            descriptor.ruleStringRecordCount,
            descriptor.ruleStringByteOffset,
            descriptor.ruleStringByteCount });
        }
        else {
            ruleStringsFinished = std::chrono::steady_clock::now();
        }

        Phase2ProjectionKeyStore sharedKeys;
        sharedKeys.arena.setPurpose(PoolPurpose::GpuHost);
        const auto appendSharedKey = [&](StrSpan key) -> uint32_t {
            ColdHashMap<BytesKeyStore, uint32_t>::InsertHint hint;
            const int32_t found = sharedKeys.offsets.lookupForInsert(key, hint);
            if (found != 0) return sharedKeys.offsets.valueAt(found);
            assert(key.len >= 0 && byteKeyBytes.size() + key.len <= capacity.byteKeyBytes);
            const uint32_t offset = static_cast<uint32_t>(byteKeyBytes.size());
            byteKeyBytes.insert(byteKeyBytes.end(), key.ptr, key.ptr + key.len);
            sharedKeys.offsets.insertAt(key, offset, hint);
            return offset;
        };

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
                const uint32_t keyOffset = appendSharedKey(key);
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
                const uint32_t keyOffset = appendSharedKey(key);
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
        if (buildPrefixFilter) {
            const auto started = std::chrono::steady_clock::now();
            appendNextPredicateMaps(descriptor, sharedKeys);
            timing.prefixFilterNanoseconds += static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - started).count());
            descriptor.byteMapViewCount = 14;
        }

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
                    && std::memcmp(blobBytes.data() + entry.keyOffset,
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
                        assert(blob.byteOffset <= blobBytes.size()
                            && blob.byteLength <= blobBytes.size() - blob.byteOffset);
                        const uint32_t keyOffset = blob.byteOffset;
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
        descriptor.podMapViewCount =
            static_cast<uint32_t>(podMapViews.size())
            - descriptor.podMapViewOffset;
        assert(descriptor.podMapViewCount == 9);

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
        const StrSpan anchorName(anchorInfo.name);
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
        descriptor.recursionAuxyImplId = body.recursionAuxyImplId;
        descriptor.recursionProductImplId = body.recursionProductImplId;
        descriptor.contradictionIndex = body.contradictionIndex;
        descriptor.mainValidityId = NameMap::MAIN_ID;
        descriptor.level = body.level;
        descriptor.standardMaxSecondaryNumber =
            parameters.standardMaxSecondaryNumber;
        if (parameters.incubator_mode)
            descriptor.evaluationFlags |= deviceEvaluationIncubatorMode;
        if (parameters.ban_disintegration)
            descriptor.evaluationFlags |= deviceEvaluationBanDisintegration;
        if (parameters.compressor_mode)
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
        timing.sourceStatementRows += static_cast<uint64_t>(statementCountSigned);
        timing.retainedStatementRows += descriptor.statementCount;
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

    /// @brief Append exact-prefix next-predicate maps to the packed projection.
    ///
    /// @details
    /// Singleton key lengths supply predicate arities. Each whole key and subkey
    /// licenses its last signed predicate after its normalized prefix. Temporary
    /// prefix and transition indexes use a private static-pool arena, with linked
    /// runs avoiding repeated insertion into one contiguous value pool. Final
    /// predicate runs use decoded-rank order and merge both signs of one core.
    ///
    /// @param block Descriptor whose ten source byte-map views are complete.
    /// @param sharedKeys This block's key-to-offset construction dictionary.
    /// @return Nothing; four bounded derived views are appended.
    /// @invariant Source key bytes and every proof-state identifier stay unchanged.
    /// @see measurePhase2ProjectionUsage, appendLogicalBlock.
    void Phase2ProjectionArena::appendNextPredicateMaps(
        const DeviceLogicalBlockProjection& block, Phase2ProjectionKeyStore& sharedKeys) {
        assert(block.byteMapViewCount == 10);
        const auto indexBytes = [&]() -> uint64_t {
            return byteMapViews.size() * sizeof(DeviceByteMapView)
                + byteMapEntries.size() * sizeof(DeviceByteMapEntry)
                + byteMapSlots.size() * sizeof(int32_t) + byteKeyBytes.size()
                + blobRecords.size() * sizeof(DeviceBlobRecord) + blobBytes.size();
        };
        const uint64_t beforeBytes = indexBytes();
        decodedRankWorkScratch_.assign(block.nameRecordCount, -1);
        const auto wordAt = [](const char* bytes, uint32_t word) {
            NameId value = 0;
            std::memcpy(&value, bytes + word * sizeof(NameId), sizeof(value));
            return value;
        };
        const auto hashBytes = [](const char* bytes, uint32_t length) {
            uint64_t hash = 14695981039346656037ull;
            for (uint32_t i = 0; i < length; ++i) {
                hash ^= static_cast<uint8_t>(bytes[i]);
                hash *= 1099511628211ull;
            }
            return hash;
        };
        for (uint32_t kind = 0; kind < 8; ++kind) {
            const DeviceByteMapView view = byteMapViews[block.byteMapViewOffset + kind];
            for (uint32_t i = 0; i < view.entryCount; ++i) {
                const DeviceByteMapEntry entry = byteMapEntries[view.entryOffset + i];
                assert(entry.keyLength >= 4 * sizeof(NameId));
                const char* bytes = byteKeyBytes.data() + entry.keyOffset;
                if (wordAt(bytes, 0) != 1) continue;
                const NameId length = wordAt(bytes, 1);
                assert(length >= 2 && length % 2 == 0);
                assert(entry.keyLength == (static_cast<uint32_t>(length) + 2) * sizeof(NameId));
                const NameId core = wordAt(bytes, 2);
                assert(core > 0 && static_cast<uint32_t>(core) < block.nameRecordCount);
                const NameId arity = (length - 2) / 2;
                assert(arity <= ExecutionParameters::MAX_ARITY);
                NameId& known = decodedRankWorkScratch_[core];
                assert(known == -1 || known == arity);
                known = arity;
            }
        }
        struct Transition {
            int32_t next;
            NameId predicate;
            uint32_t mask;
            int16_t bound[ExecutionParameters::MAX_ARITY];
        };
        for (uint32_t memory = 0; memory < 4; ++memory) {
            LbArena scratch(&gpuHostStorage());
            scratch.setPurpose(PoolPurpose::GpuHost);  // pool attribution
            DirtyState dirty = DirtyState::Clean;
            ColdHashSet<BytesKeyStore> prefixes(&scratch, &dirty);
            PagedVector<int32_t> heads(&scratch, &dirty);
            PagedVector<Transition> transitions(&scratch, &dirty);
            ColdHashMap<PodKeyStore<uint64_t>, int32_t> transitionIds(&scratch, &dirty);
            for (uint32_t half = 0; half < 2; ++half) {
                const DeviceByteMapView view = byteMapViews[
                    block.byteMapViewOffset + 2 * memory + half];
                for (uint32_t i = 0; i < view.entryCount; ++i) {
                    const DeviceByteMapEntry entry = byteMapEntries[view.entryOffset + i];
                    const char* bytes = byteKeyBytes.data() + entry.keyOffset;
                    const NameId count = wordAt(bytes, 0);
                    const NameId length = wordAt(bytes, 1);
                    assert(count > 0 && count <= ExecutionParameters::MAX_EXPRESSIONS);
                    assert(length >= 2 * count && length <= ExecutionParameters::MAX_KEY_SLOTS);
                    assert(entry.keyLength == (static_cast<uint32_t>(length) + 2) * sizeof(NameId));
                    uint32_t prefixLength = 0;
                    NameId prefixVariables = 0;
                    for (NameId premise = 0; premise < count - 1; ++premise) {
                        assert(prefixLength + 2 <= static_cast<uint32_t>(length));
                        const NameId core = wordAt(bytes, 2 + prefixLength);
                        assert(core > 0 && static_cast<uint32_t>(core) < block.nameRecordCount);
                        const NameId arity = decodedRankWorkScratch_[core];
                        assert(arity >= 0 && "every normalized premise has a singleton subkey");
                        for (NameId argument = 0; argument < arity; ++argument) {
                            const NameId variable = wordAt(bytes,
                                4 + prefixLength + 2 * static_cast<uint32_t>(argument));
                            if (variable > prefixVariables) prefixVariables = variable;
                        }
                        prefixLength += 2 + 2 * static_cast<uint32_t>(arity);
                    }
                    assert(prefixLength + 2 <= static_cast<uint32_t>(length));
                    const NameId core = wordAt(bytes, 2 + prefixLength);
                    const NameId polarity = wordAt(bytes, 3 + prefixLength);
                    assert(core > 0 && static_cast<uint32_t>(core) < block.nameRecordCount);
                    assert(polarity == 0 || polarity == 1);
                    const NameId arity = decodedRankWorkScratch_[core];
                    assert(arity >= 0);
                    assert(prefixLength + 2 + 2 * static_cast<uint32_t>(arity)
                        == static_cast<uint32_t>(length));
                    NameId nextArguments[ExecutionParameters::MAX_ARITY];
                    for (NameId argument = 0; argument < arity; ++argument)
                        nextArguments[argument] = wordAt(bytes,
                            4 + prefixLength + 2 * static_cast<uint32_t>(argument));
                    NameId key[2 + ExecutionParameters::MAX_KEY_SLOTS];
                    key[0] = count - 1;
                    key[1] = static_cast<NameId>(prefixLength);
                    std::memcpy(key + 2, bytes + 2 * sizeof(NameId), prefixLength * sizeof(NameId));
                    const int32_t prefix = prefixes.mint(StrSpan(
                        reinterpret_cast<const char*>(key),
                        static_cast<int32_t>((2 + prefixLength) * sizeof(NameId))));
                    if (prefix > heads.size()) {
                        assert(prefix == heads.size() + 1);
                        heads.push_back(0);
                    }
                    const uint64_t identity = (static_cast<uint64_t>(prefix) << 32)
                        | static_cast<uint32_t>(core);
                    const int32_t existing = transitionIds.lookup(identity);
                    if (existing != 0) {
                        const int32_t index = transitionIds.valueAt(existing) - 1;
                        Transition value = transitions[index];
                        value.mask |= 1u << polarity;
                        mergeNextPremiseBindings(value.bound, arity, nextArguments,
                            prefixVariables, false);
                        transitions.setAt(index, value);
                    }
                    else {
                        Transition value{heads[prefix - 1], core, 1u << polarity, {}};
                        mergeNextPremiseBindings(value.bound, arity, nextArguments,
                            prefixVariables, true);
                        transitions.push_back(value);
                        const int32_t reference = transitions.size();
                        heads.setAt(prefix - 1, reference);
                        transitionIds.insert(identity, reference);
                    }
                }
            }
            DeviceByteMapView view{};
            view.entryOffset = static_cast<uint32_t>(byteMapEntries.size());
            view.entryCount = static_cast<uint32_t>(prefixes.count());
            timing.prefixFilterKeys += view.entryCount;
            timing.prefixFilterPredicates += static_cast<uint64_t>(transitions.size());
            view.slotOffset = static_cast<uint32_t>(byteMapSlots.size());
            view.kind = static_cast<DeviceByteMapKind>(10 + memory);
            if (view.entryCount != 0) {
                view.slotCount = 2;
                while (static_cast<uint64_t>(view.slotCount) < 2ull * view.entryCount) {
                    assert(view.slotCount <= (UINT32_MAX >> 1));
                    view.slotCount <<= 1;
                }
            }
            assert(byteMapViews.size() < capacity.byteMapViews);
            assert(byteMapEntries.size() + view.entryCount <= capacity.byteMapEntries);
            assert(byteMapSlots.size() + view.slotCount <= capacity.byteMapSlots);
            byteMapSlots.resize(byteMapSlots.size() + view.slotCount, -1);
            for (int32_t prefix = 1; prefix <= prefixes.count(); ++prefix) {
                const StrSpan key = prefixes.keyAt(prefix);
                assert(key.len >= 2 * static_cast<int32_t>(sizeof(NameId)));
                ColdHashMap<BytesKeyStore, uint32_t>::InsertHint hint;
                const int32_t found = sharedKeys.offsets.lookupForInsert(key, hint);
                uint32_t keyOffset;
                if (found != 0) {
                    keyOffset = sharedKeys.offsets.valueAt(found);
                }
                else {
                    assert(byteKeyBytes.size() + key.len <= capacity.byteKeyBytes);
                    keyOffset = static_cast<uint32_t>(byteKeyBytes.size());
                    byteKeyBytes.insert(byteKeyBytes.end(), key.ptr, key.ptr + key.len);
                    sharedKeys.offsets.insertAt(key, keyOffset, hint);
                }
                decodedRankScratch_.clear();
                for (int32_t reference = heads[prefix - 1]; reference != 0;
                     reference = transitions[reference - 1].next) {
                    assert(decodedRankScratch_.size() < capacity.nameRecords);
                    decodedRankScratch_.push_back(reference - 1);
                }
                assert(!decodedRankScratch_.empty());
                std::sort(decodedRankScratch_.begin(), decodedRankScratch_.end(),
                    [&](NameId left, NameId right) {
                        return nameRecords[block.nameRecordOffset + transitions[left].predicate].decodedLexRank
                            < nameRecords[block.nameRecordOffset + transitions[right].predicate].decodedLexRank;
                    });
                const uint32_t valueBytes = static_cast<uint32_t>(decodedRankScratch_.size()) * 8;
                assert(blobRecords.size() < capacity.blobRecords);
                assert(blobBytes.size() + valueBytes <= capacity.blobBytes);
                const uint32_t record = static_cast<uint32_t>(blobRecords.size());
                const uint32_t valueOffset = static_cast<uint32_t>(blobBytes.size());
                blobBytes.resize(blobBytes.size() + valueBytes);
                uint32_t at = valueOffset;
                for (NameId index : decodedRankScratch_) {
                    const Transition value = transitions[index];
                    int32_t gateSlots[2]{ -1, -1 };
                    NameId gateVariables[2]{ 0, 0 };
                    int32_t gates = 0;
                    for (int32_t slot = 0; slot < ExecutionParameters::MAX_ARITY && gates < 2; ++slot) {
                        if (value.bound[slot] <= 0) continue;
                        gateSlots[gates] = slot;
                        gateVariables[gates] = value.bound[slot];
                        ++gates;
                    }
                    const uint32_t word = packNextPredicateWord(value.mask,
                        gateSlots[0], gateVariables[0], gateSlots[1], gateVariables[1]);
                    std::memcpy(blobBytes.data() + at, &value.predicate, 4);
                    std::memcpy(blobBytes.data() + at + 4, &word, 4);
                    at += 8;
                }
                blobRecords.push_back(DeviceBlobRecord{valueOffset, valueBytes});
                byteMapEntries.push_back(DeviceByteMapEntry{
                    keyOffset, static_cast<uint32_t>(key.len), record, 1});
                uint32_t slot = static_cast<uint32_t>(hashBytes(
                    byteKeyBytes.data() + keyOffset, static_cast<uint32_t>(key.len)))
                    & (view.slotCount - 1);
                while (byteMapSlots[view.slotOffset + slot] != -1)
                    slot = (slot + 1) & (view.slotCount - 1);
                byteMapSlots[view.slotOffset + slot] = static_cast<int32_t>(byteMapEntries.size() - 1);
            }
            byteMapViews.push_back(view);
        }
        timing.prefixFilterBytes += indexBytes() - beforeBytes;
    }

    /// @brief Translate a retained host statement ordinal to its device ordinal.
    ///
    /// @details
    /// Binary search uses the strictly ascending source ordinals in the block's
    /// slice. Shard concatenation preserves these local ordinals unchanged.
    ///
    /// @param blockIndex Logical-block descriptor index in this arena.
    /// @param sourceIndex Original overall-list ordinal of a retained premise.
    /// @return The corresponding block-local device statement ordinal.
    /// @invariant The source statement survives the signed-core prefilter.
    /// @see Phase2ProjectionArena::appendLogicalBlock.
    NameId Phase2ProjectionArena::projectedStatementIndex(
        uint32_t blockIndex, NameId sourceIndex) const {
        assert(blockIndex < logicalBlocks.size());
        const auto& block = logicalBlocks[blockIndex];
        const auto begin = sourceStatementIndices.begin() + block.statementOffset;
        const auto end = begin + block.statementCount;
        const auto found = std::lower_bound(begin, end, sourceIndex);
        assert(found != end && *found == sourceIndex
            && "processor stump premise was excluded from the device projection");
        return static_cast<NameId>(found - begin);
    }


    /// @brief Compute the column ends following one construction window.
    ///
    /// @details Reuses checked shard sizing without reading or changing payload.
    /// @param bases Column ends before this window.
    /// @return Column ends after this window.
    Phase2ProjectionArena::ShardBases Phase2ProjectionArena::streamedEnd(
        const ShardBases& bases) const { return followingShardBases(bases, *this); }

    /// @brief Rebase a complete construction window for an append upload.
    ///
    /// @details Uses ordinary adoption's per-column transformation; no storage
    /// is allocated or released and used lengths remain unchanged.
    /// @param bases Assigned complete-image column offsets.
    /// @return Nothing.
    void Phase2ProjectionArena::rebaseForStream(const ShardBases& bases) {
        assert(role_ != Role::Sealing);
        for (uint32_t column = 0; column < kShardColumnCount; ++column)
            rebaseColumn(column, 0, bases);
    }

    /// @brief Retain uploaded lookup data required for host-side sealing.
    ///
    /// @details Source ordinals support later task construction. Every copied
    /// descriptor already addresses its complete-image column; the device owns
    /// all remaining uploaded data independently of the construction window.
    /// @param source Rebased window whose synchronous upload has completed.
    /// @param bases Assigned complete-image offsets of the window.
    /// @return Nothing.
    void Phase2ProjectionArena::appendSealingProjection(
        const Phase2ProjectionArena& source, const ShardBases& bases) {
        assert(role_ == Role::Sealing);
        const auto append = [](auto& destination, const auto& input, uint32_t base) {
            assert(destination.size() == base);
            destination.insert(destination.end(), input.begin(), input.end());
        };
        append(logicalBlocks, source.logicalBlocks, bases.block);
        append(sourceStatementIndices, source.sourceStatementIndices, bases.statement);
        append(nameRecords, source.nameRecords, bases.nameRecord);
        append(nameBytes, source.nameBytes, bases.nameByte);
        append(ruleStringRecords, source.ruleStringRecords, bases.ruleRecord);
        append(ruleStringBytes, source.ruleStringBytes, bases.ruleByte);
        addShardTiming(source);
        timing.logicalBlocks = logicalBlocks.size();
        assert(timing.mergeNanoseconds <= std::numeric_limits<uint64_t>::max()
            - source.timing.mergeNanoseconds);
        timing.mergeNanoseconds += source.timing.mergeNanoseconds;
    }

    /// @brief The destination column sizes before a shard is appended — the
    ///        bases its rebased references add.
    ///
    /// @details
    /// Reads the current size of every destination column, each asserted to fit
    /// `uint32_t` (every projected offset is 32-bit).
    ///
    /// @return The bases of the next shard appended to this arena.
    Phase2ProjectionArena::ShardBases Phase2ProjectionArena::currentShardBases() const {
        const auto checkedBase = [](std::size_t size) -> uint32_t {
            assert(size <= std::numeric_limits<uint32_t>::max());
            return static_cast<uint32_t>(size);
        };
        ShardBases bases{};
        bases.block = checkedBase(logicalBlocks.size());
        bases.statement = checkedBase(statements.size());
        bases.nameRecord = checkedBase(nameRecords.size());
        bases.nameByte = checkedBase(nameBytes.size());
        bases.nameSlot = checkedBase(nameSlots.size());
        bases.ruleRecord = checkedBase(ruleStringRecords.size());
        bases.ruleByte = checkedBase(ruleStringBytes.size());
        bases.byteView = checkedBase(byteMapViews.size());
        bases.byteEntry = checkedBase(byteMapEntries.size());
        bases.byteSlot = checkedBase(byteMapSlots.size());
        bases.byteKey = checkedBase(byteKeyBytes.size());
        bases.blobRecord = checkedBase(blobRecords.size());
        bases.blobByte = checkedBase(blobBytes.size());
        bases.reverseView = checkedBase(reverseMapViews.size());
        bases.reverseEntry = checkedBase(reverseMapEntries.size());
        bases.reverseSlot = checkedBase(reverseMapSlots.size());
        bases.reverseKey = checkedBase(reverseKeyBytes.size());
        bases.reverseOwner = checkedBase(reverseOwners.size());
        bases.podView = checkedBase(podMapViews.size());
        bases.podEntry = checkedBase(podMapEntries.size());
        bases.podSlot = checkedBase(podMapSlots.size());
        bases.podRun = checkedBase(podRunValues.size());
        bases.mandatoryKey = checkedBase(mandatoryStatementKeys.size());
        bases.metadata = checkedBase(metadataBytes.size());
        return bases;
    }

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
    Phase2ProjectionArena::ShardBases Phase2ProjectionArena::followingShardBases(
        const ShardBases& bases, const Phase2ProjectionArena& source) {
        const auto add = [](uint32_t base, std::size_t size) -> uint32_t {
            assert(size <= std::numeric_limits<uint32_t>::max() - base);
            return base + static_cast<uint32_t>(size);
        };
        ShardBases next{};
        next.block = add(bases.block, source.logicalBlocks.size());
        next.statement = add(bases.statement, source.statements.size());
        next.nameRecord = add(bases.nameRecord, source.nameRecords.size());
        next.nameByte = add(bases.nameByte, source.nameBytes.size());
        next.nameSlot = add(bases.nameSlot, source.nameSlots.size());
        next.ruleRecord = add(bases.ruleRecord, source.ruleStringRecords.size());
        next.ruleByte = add(bases.ruleByte, source.ruleStringBytes.size());
        next.byteView = add(bases.byteView, source.byteMapViews.size());
        next.byteEntry = add(bases.byteEntry, source.byteMapEntries.size());
        next.byteSlot = add(bases.byteSlot, source.byteMapSlots.size());
        next.byteKey = add(bases.byteKey, source.byteKeyBytes.size());
        next.blobRecord = add(bases.blobRecord, source.blobRecords.size());
        next.blobByte = add(bases.blobByte, source.blobBytes.size());
        next.reverseView = add(bases.reverseView, source.reverseMapViews.size());
        next.reverseEntry = add(bases.reverseEntry, source.reverseMapEntries.size());
        next.reverseSlot = add(bases.reverseSlot, source.reverseMapSlots.size());
        next.reverseKey = add(bases.reverseKey, source.reverseKeyBytes.size());
        next.reverseOwner = add(bases.reverseOwner, source.reverseOwners.size());
        next.podView = add(bases.podView, source.podMapViews.size());
        next.podEntry = add(bases.podEntry, source.podMapEntries.size());
        next.podSlot = add(bases.podSlot, source.podMapSlots.size());
        next.podRun = add(bases.podRun, source.podRunValues.size());
        next.mandatoryKey = add(bases.mandatoryKey, source.mandatoryStatementKeys.size());
        next.metadata = add(bases.metadata, source.metadataBytes.size());
        return next;
    }

    /// @brief Assert that `source` adopted at `bases` fits every declared
    ///        ceiling (`capacity`), the size the device buffer is built for.
    ///
    /// @param bases Bases at which `source` is adopted.
    /// @param source The shard to adopt.
    /// @return Nothing.
    void Phase2ProjectionArena::assertShardFits(const ShardBases& bases,
        const Phase2ProjectionArena& source) const {
        assert(bases.block + source.logicalBlocks.size() <= capacity.logicalBlocks);
        assert(bases.statement + source.statements.size() <= capacity.statements);
        assert(bases.nameRecord + source.nameRecords.size() <= capacity.nameRecords);
        assert(bases.nameByte + source.nameBytes.size() <= capacity.nameBytes);
        assert(bases.nameSlot + source.nameSlots.size() <= capacity.nameSlots);
        assert(bases.ruleRecord + source.ruleStringRecords.size()
            <= capacity.ruleStringRecords);
        assert(bases.ruleByte + source.ruleStringBytes.size()
            <= capacity.ruleStringBytes);
        assert(bases.byteView + source.byteMapViews.size() <= capacity.byteMapViews);
        assert(bases.byteEntry + source.byteMapEntries.size()
            <= capacity.byteMapEntries);
        assert(bases.byteSlot + source.byteMapSlots.size() <= capacity.byteMapSlots);
        assert(bases.byteKey + source.byteKeyBytes.size() <= capacity.byteKeyBytes);
        assert(bases.blobRecord + source.blobRecords.size() <= capacity.blobRecords);
        assert(bases.blobByte + source.blobBytes.size() <= capacity.blobBytes);
        assert(bases.reverseView + source.reverseMapViews.size()
            <= capacity.reverseMapViews);
        assert(bases.reverseEntry + source.reverseMapEntries.size()
            <= capacity.reverseMapEntries);
        assert(bases.reverseSlot + source.reverseMapSlots.size()
            <= capacity.reverseMapSlots);
        assert(bases.reverseKey + source.reverseKeyBytes.size()
            <= capacity.reverseKeyBytes);
        assert(bases.reverseOwner + source.reverseOwners.size()
            <= capacity.reverseOwners);
        assert(bases.podView + source.podMapViews.size() <= capacity.podMapViews);
        assert(bases.podEntry + source.podMapEntries.size() <= capacity.podMapEntries);
        assert(bases.podSlot + source.podMapSlots.size() <= capacity.podMapSlots);
        assert(bases.podRun + source.podRunValues.size() <= capacity.podRunValues);
        assert(bases.mandatoryKey + source.mandatoryStatementKeys.size()
            <= capacity.mandatoryStatementKeys);
        assert(bases.metadata + source.metadataBytes.size() <= capacity.metadataBytes);
    }

    /// @brief Move one column of `shard` to the end of this arena's column and
    ///        rebase the moved references by `referenceBases`.
    ///
    /// @details
    /// The per-column core of `adoptShards`: column `column`
    /// (0 .. `kShardColumnCount` - 1) is the only destination column written, and
    /// every rebase reads only the two bases, so different columns may be adopted
    /// by different threads. The column must end at its base in `bases` when this
    /// is called (shards are adopted in order).
    ///
    /// @param column Column ordinal.
    /// @param shard The slice shard whose column is moved; the column ends empty.
    /// @param bases Positions in this arena at which `shard` is placed.
    /// @param referenceBases Offsets the moved references add: `bases` plus this
    ///        arena's complete-image stream bases.
    /// @return Nothing.
    /// @invariant Every rebased reference addresses the byte-identical moved
    ///            value.
    void Phase2ProjectionArena::adoptShardColumn(uint32_t column,
        Phase2ProjectionArena& shard, const ShardBases& bases,
        const ShardBases& referenceBases) {
        const auto place = [](auto& destination, auto& source, uint32_t base) {
            assert(destination.size() == base);
            destination.appendMoved(source);
        };
        uint32_t begin = 0;
        switch (column) {
        case 0: begin = bases.block; place(logicalBlocks, shard.logicalBlocks, begin); break;
        case 1: begin = bases.statement; place(statements, shard.statements, begin); break;
        case 2: begin = bases.statement; place(sourceStatementIndices, shard.sourceStatementIndices, begin); break;
        case 3: begin = bases.nameRecord; place(nameRecords, shard.nameRecords, begin); break;
        case 4: begin = bases.nameByte; place(nameBytes, shard.nameBytes, begin); break;
        case 5: begin = bases.nameSlot; place(nameSlots, shard.nameSlots, begin); break;
        case 6: begin = bases.ruleRecord; place(ruleStringRecords, shard.ruleStringRecords, begin); break;
        case 7: begin = bases.ruleByte; place(ruleStringBytes, shard.ruleStringBytes, begin); break;
        case 8: begin = bases.byteView; place(byteMapViews, shard.byteMapViews, begin); break;
        case 9: begin = bases.byteEntry; place(byteMapEntries, shard.byteMapEntries, begin); break;
        case 10: begin = bases.byteSlot; place(byteMapSlots, shard.byteMapSlots, begin); break;
        case 11: begin = bases.byteKey; place(byteKeyBytes, shard.byteKeyBytes, begin); break;
        case 12: begin = bases.blobRecord; place(blobRecords, shard.blobRecords, begin); break;
        case 13: begin = bases.blobByte; place(blobBytes, shard.blobBytes, begin); break;
        case 14: begin = bases.reverseView; place(reverseMapViews, shard.reverseMapViews, begin); break;
        case 15: begin = bases.reverseEntry; place(reverseMapEntries, shard.reverseMapEntries, begin); break;
        case 16: begin = bases.reverseSlot; place(reverseMapSlots, shard.reverseMapSlots, begin); break;
        case 17: begin = bases.reverseKey; place(reverseKeyBytes, shard.reverseKeyBytes, begin); break;
        case 18: begin = bases.reverseOwner; place(reverseOwners, shard.reverseOwners, begin); break;
        case 19: begin = bases.podView; place(podMapViews, shard.podMapViews, begin); break;
        case 20: begin = bases.podEntry; place(podMapEntries, shard.podMapEntries, begin); break;
        case 21: begin = bases.podSlot; place(podMapSlots, shard.podMapSlots, begin); break;
        case 22: begin = bases.podRun; place(podRunValues, shard.podRunValues, begin); break;
        case 23: begin = bases.mandatoryKey; place(mandatoryStatementKeys, shard.mandatoryStatementKeys, begin); break;
        case 24: begin = bases.metadata; place(metadataBytes, shard.metadataBytes, begin); break;
        default: assert(false && "unknown projection column");
        }
        rebaseColumn(column, begin, referenceBases);
    }

    /// @brief Rebase references in one newly placed host column range.
    ///
    /// @details Both placement paths use this exact offset conversion. Payload
    /// values and identifiers local to a logical block are unchanged.
    /// @param column Host column ordinal.
    /// @param begin First newly placed row in that column.
    /// @param bases Assigned bases of every referenced device column.
    /// @return Nothing.
    void Phase2ProjectionArena::rebaseColumn(uint32_t column, uint32_t begin,
        const ShardBases& bases) {
        const auto checkedAdd = [](uint32_t value, uint32_t base) -> uint32_t {
            assert(value <= std::numeric_limits<uint32_t>::max() - base);
            return value + base;
        };
        const auto shiftSlots = [](auto& slots, uint32_t begin,
            uint32_t entryBase) {
            for (std::size_t index = begin; index < slots.size(); ++index)
                if (slots[index] >= 0) slots[index] += static_cast<int32_t>(entryBase);
        };
        switch (column) {
        case 0:
            for (std::size_t index = begin; index < logicalBlocks.size(); ++index) {
                DeviceLogicalBlockProjection& descriptor = logicalBlocks[index];
                descriptor.statementOffset = checkedAdd(
                    descriptor.statementOffset, bases.statement);
                descriptor.nameRecordOffset = checkedAdd(
                    descriptor.nameRecordOffset, bases.nameRecord);
                descriptor.nameByteOffset = checkedAdd(
                    descriptor.nameByteOffset, bases.nameByte);
                descriptor.nameSlotOffset = checkedAdd(
                    descriptor.nameSlotOffset, bases.nameSlot);
                descriptor.ruleStringRecordOffset = checkedAdd(
                    descriptor.ruleStringRecordOffset, bases.ruleRecord);
                descriptor.ruleStringByteOffset = checkedAdd(
                    descriptor.ruleStringByteOffset, bases.ruleByte);
                descriptor.byteMapViewOffset = checkedAdd(
                    descriptor.byteMapViewOffset, bases.byteView);
                descriptor.reverseMapViewOffset = checkedAdd(
                    descriptor.reverseMapViewOffset, bases.reverseView);
                descriptor.podMapViewOffset = checkedAdd(
                    descriptor.podMapViewOffset, bases.podView);
                descriptor.localKeyOffset = checkedAdd(
                    descriptor.localKeyOffset, bases.mandatoryKey);
                descriptor.deltaKeyOffset = checkedAdd(
                    descriptor.deltaKeyOffset, bases.mandatoryKey);
                descriptor.externalKeyOffset = checkedAdd(
                    descriptor.externalKeyOffset, bases.mandatoryKey);
                descriptor.metadataOffset = checkedAdd(
                    descriptor.metadataOffset, bases.metadata);
                descriptor.anchorNameOffset = checkedAdd(
                    descriptor.anchorNameOffset, bases.metadata);
            }
            break;
        case 1: break;
        case 2:
            break;
        case 3:
            for (std::size_t index = begin; index < nameRecords.size(); ++index)
                nameRecords[index].byteOffset = checkedAdd(
                    nameRecords[index].byteOffset, bases.nameByte);
            break;
        case 4: break;
        case 5:
            shiftSlots(nameSlots, begin, bases.nameRecord);
            break;
        case 6:
            for (std::size_t index = begin; index < ruleStringRecords.size(); ++index)
                ruleStringRecords[index].byteOffset = checkedAdd(
                    ruleStringRecords[index].byteOffset, bases.ruleByte);
            break;
        case 7: break;
        case 8:
            for (std::size_t index = begin; index < byteMapViews.size(); ++index) {
                byteMapViews[index].entryOffset = checkedAdd(
                    byteMapViews[index].entryOffset, bases.byteEntry);
                byteMapViews[index].slotOffset = checkedAdd(
                    byteMapViews[index].slotOffset, bases.byteSlot);
            }
            break;
        case 9:
            for (std::size_t index = begin; index < byteMapEntries.size(); ++index) {
                byteMapEntries[index].keyOffset = checkedAdd(
                    byteMapEntries[index].keyOffset, bases.byteKey);
                byteMapEntries[index].blobRecordOffset = checkedAdd(
                    byteMapEntries[index].blobRecordOffset, bases.blobRecord);
            }
            break;
        case 10:
            shiftSlots(byteMapSlots, begin, bases.byteEntry);
            break;
        case 11: break;
        case 12:
            for (std::size_t index = begin; index < blobRecords.size(); ++index)
                blobRecords[index].byteOffset = checkedAdd(
                    blobRecords[index].byteOffset, bases.blobByte);
            break;
        case 13: break;
        case 14:
            for (std::size_t index = begin; index < reverseMapViews.size(); ++index) {
                reverseMapViews[index].entryOffset = checkedAdd(
                    reverseMapViews[index].entryOffset, bases.reverseEntry);
                reverseMapViews[index].slotOffset = checkedAdd(
                    reverseMapViews[index].slotOffset, bases.reverseSlot);
            }
            break;
        case 15:
            for (std::size_t index = begin; index < reverseMapEntries.size(); ++index) {
                reverseMapEntries[index].keyOffset = checkedAdd(
                    reverseMapEntries[index].keyOffset, bases.blobByte);
                reverseMapEntries[index].ownerOffset = checkedAdd(
                    reverseMapEntries[index].ownerOffset, bases.reverseOwner);
            }
            break;
        case 16:
            shiftSlots(reverseMapSlots, begin, bases.reverseEntry);
            break;
        case 17: break;
        case 18: break;
        case 19:
            for (std::size_t index = begin; index < podMapViews.size(); ++index) {
                podMapViews[index].entryOffset = checkedAdd(
                    podMapViews[index].entryOffset, bases.podEntry);
                podMapViews[index].slotOffset = checkedAdd(
                    podMapViews[index].slotOffset, bases.podSlot);
            }
            break;
        case 20:
            for (std::size_t index = begin; index < podMapEntries.size(); ++index)
                podMapEntries[index].runOffset = checkedAdd(
                    podMapEntries[index].runOffset, bases.podRun);
            break;
        case 21:
            shiftSlots(podMapSlots, begin, bases.podEntry);
            break;
        case 22: break;
        case 23:
            break;
        case 24: break;
        default:
            assert(false && "Phase2ProjectionArena::rebaseColumn: unknown column");
        }
    }

    /// @brief Add a shard's construction timing to this arena's totals.
    ///
    /// @param source The appended shard.
    /// @return Nothing.
    void Phase2ProjectionArena::addShardTiming(const Phase2ProjectionArena& source) {
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
        addCounter(timing.prefixFilterNanoseconds, source.timing.prefixFilterNanoseconds);
        addCounter(timing.prefixFilterKeys, source.timing.prefixFilterKeys);
        addCounter(timing.prefixFilterPredicates, source.timing.prefixFilterPredicates);
        addCounter(timing.prefixFilterBytes, source.timing.prefixFilterBytes);
        addCounter(timing.reverseMapNanoseconds,
            source.timing.reverseMapNanoseconds);
        addCounter(timing.podMapNanoseconds,
            source.timing.podMapNanoseconds);
        addCounter(timing.finalNanoseconds,
            source.timing.finalNanoseconds);
        addCounter(timing.sourceStatementRows, source.timing.sourceStatementRows);
        addCounter(timing.retainedStatementRows, source.timing.retainedStatementRows);
    }

    /// @brief Take in the rows this backing arena's slice shards built, in place,
    ///        with every reference at its complete-image offset.
    ///
    /// @details
    /// Computes every shard's bases up front (`currentShardBases`, then
    /// `followingShardBases` shard by shard), asserts the totals fit `capacity`,
    /// adds `streamBases` to obtain each shard's reference bases, and lets the
    /// persistent `workers` take whole columns: a column moves each shard's used
    /// prefix to the end of this arena's column and rebases the moved references,
    /// shard by shard in order (`adoptShardColumn`). Moving stays inside the
    /// backing storage, so the projection exists once
    /// (D-425); adding the stream bases here
    /// replaces a separate rebase pass over a streamed window
    /// (D-447). The result is byte-identical
    /// to building every block in one arena and rebasing it by `streamBases`; the
    /// timing totals match and the merge time is counted once.
    ///
    /// @param shards This arena's slice shards in slice order.
    /// @param count Number of shards; at most `kMaxAppendedShards`.
    /// @param firstBlocks Output: first logical-block index of each shard.
    /// @param streamBases Complete-image offset of this arena's first row in every
    ///        column; each sum with a placed base is checked to fit `uint32_t`.
    /// @param workers Persistent threads that run the column tasks.
    /// @return Nothing.
    /// @invariant Every rebased reference addresses the byte-identical moved
    ///            value at its complete-image position, and no column reallocates.
    void Phase2ProjectionArena::adoptShards(Phase2ProjectionArena* const* shards,
        uint32_t count, uint32_t* firstBlocks, const ShardBases& streamBases,
        Phase2ProjectionWorkers& workers) {
        const auto mergeStarted = std::chrono::steady_clock::now();
        assert(shards != nullptr && firstBlocks != nullptr);
        assert(role_ == Role::Backing && !adopted_);
        assert(count <= kMaxAppendedShards);
        adopted_ = true;
        const auto streamed = [&streamBases](const ShardBases& placed) {
            const auto add = [](uint32_t position, uint32_t stream) -> uint32_t {
                assert(position <= std::numeric_limits<uint32_t>::max() - stream
                    && "streamed adoption offset leaves uint32_t");
                return position + stream;
            };
            ShardBases reference{};
            reference.block = add(placed.block, streamBases.block);
            reference.statement = add(placed.statement, streamBases.statement);
            reference.nameRecord = add(placed.nameRecord, streamBases.nameRecord);
            reference.nameByte = add(placed.nameByte, streamBases.nameByte);
            reference.nameSlot = add(placed.nameSlot, streamBases.nameSlot);
            reference.ruleRecord = add(placed.ruleRecord, streamBases.ruleRecord);
            reference.ruleByte = add(placed.ruleByte, streamBases.ruleByte);
            reference.byteView = add(placed.byteView, streamBases.byteView);
            reference.byteEntry = add(placed.byteEntry, streamBases.byteEntry);
            reference.byteSlot = add(placed.byteSlot, streamBases.byteSlot);
            reference.byteKey = add(placed.byteKey, streamBases.byteKey);
            reference.blobRecord = add(placed.blobRecord, streamBases.blobRecord);
            reference.blobByte = add(placed.blobByte, streamBases.blobByte);
            reference.reverseView = add(placed.reverseView, streamBases.reverseView);
            reference.reverseEntry = add(placed.reverseEntry, streamBases.reverseEntry);
            reference.reverseSlot = add(placed.reverseSlot, streamBases.reverseSlot);
            reference.reverseKey = add(placed.reverseKey, streamBases.reverseKey);
            reference.reverseOwner = add(placed.reverseOwner, streamBases.reverseOwner);
            reference.podView = add(placed.podView, streamBases.podView);
            reference.podEntry = add(placed.podEntry, streamBases.podEntry);
            reference.podSlot = add(placed.podSlot, streamBases.podSlot);
            reference.podRun = add(placed.podRun, streamBases.podRun);
            reference.mandatoryKey = add(placed.mandatoryKey, streamBases.mandatoryKey);
            reference.metadata = add(placed.metadata, streamBases.metadata);
            return reference;
        };
        std::array<ShardBases, kMaxAppendedShards> bases{};
        std::array<ShardBases, kMaxAppendedShards> referenceBases{};
        ShardBases next = currentShardBases();
        for (uint32_t shard = 0; shard < count; ++shard) {
            assert(shards[shard] != nullptr && shards[shard]->role_ == Role::Slice);
            bases[shard] = next;
            referenceBases[shard] = streamed(next);
            assertShardFits(next, *shards[shard]);
            firstBlocks[shard] = next.block;
            next = followingShardBases(next, *shards[shard]);
        }
        auto columnJob = [&](uint32_t column) {
            for (uint32_t shard = 0; shard < count; ++shard)
                adoptShardColumn(column, *shards[shard], bases[shard], referenceBases[shard]);
        };
        workers.run(kShardColumnCount, columnJob);
        for (uint32_t shard = 0; shard < count; ++shard) addShardTiming(*shards[shard]);
        timing.logicalBlocks = logicalBlocks.size();
        const int64_t elapsed = std::chrono::duration_cast<
            std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - mergeStarted).count();
        assert(elapsed >= 0);
        assert(timing.mergeNanoseconds
            <= std::numeric_limits<uint64_t>::max() - static_cast<uint64_t>(elapsed));
        timing.mergeNanoseconds += static_cast<uint64_t>(elapsed);
    }

    /// @brief Start the persistent threads.
    ///
    /// @details
    /// The threads wait for the first job. Nothing else is allocated later:
    /// jobs reuse the same threads, mutex and condition variables.
    ///
    /// @param threadCount Number of threads; positive.
    /// @return Idle workers.
    Phase2ProjectionWorkers::Phase2ProjectionWorkers(uint32_t threadCount)
        : threads_(threadCount) {
        assert(threadCount > 0);
        for (uint32_t index = 0; index < threadCount; ++index)
            threads_[index] = std::thread([this] { work(); });
    }

    /// @brief Stop and join every thread.
    ///
    /// @details
    /// Runs only between jobs (`run` returns after its job), so no task is in
    /// flight; each thread observes the stop request and exits.
    ///
    /// @return Nothing.
    Phase2ProjectionWorkers::~Phase2ProjectionWorkers() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            assert(workingThreads_ == 0 && !stopping_);
            stopping_ = true;
        }
        start_.notify_all();
        for (std::size_t index = 0; index < threads_.size(); ++index) threads_[index].join();
    }

    /// @brief Publish one job to the threads and wait for its last task.
    ///
    /// @details
    /// Under the mutex stores the job, resets the task counter and the count of
    /// threads still working, advances the generation that wakes every thread,
    /// then waits until each thread has finished claiming tasks.
    ///
    /// @param taskCount Number of task indices.
    /// @param invoke Type-erased call of the job.
    /// @param job The caller's job object.
    /// @return Nothing.
    /// @invariant No other job is in flight and the workers are not stopping.
    void Phase2ProjectionWorkers::dispatch(uint32_t taskCount,
        void (*invoke)(void*, uint32_t), void* job) {
        assert(invoke != nullptr && job != nullptr);
        std::unique_lock<std::mutex> lock(mutex_);
        assert(workingThreads_ == 0 && !stopping_
            && "projection workers run one job at a time");
        invoke_ = invoke;
        job_ = job;
        taskCount_ = taskCount;
        nextTask_.store(0, std::memory_order_relaxed);
        workingThreads_ = static_cast<uint32_t>(threads_.size());
        ++generation_;
        start_.notify_all();
        finished_.wait(lock, [this] { return workingThreads_ == 0; });
    }

    /// @brief Thread body: wait for a new generation, claim its tasks, report.
    ///
    /// @details
    /// Each thread runs claimed tasks until the counter passes `taskCount`, then
    /// decrements the count of working threads; the last one wakes the
    /// dispatcher. A stop request with no new generation ends the thread.
    ///
    /// @return Nothing.
    void Phase2ProjectionWorkers::work() {
        uint64_t seen = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            start_.wait(lock, [this, &seen] { return stopping_ || generation_ != seen; });
            if (generation_ == seen) return;
            seen = generation_;
            void (*const invoke)(void*, uint32_t) = invoke_;
            void* const job = job_;
            const uint32_t taskCount = taskCount_;
            lock.unlock();
            for (uint32_t task = nextTask_.fetch_add(1, std::memory_order_relaxed);
                 task < taskCount;
                 task = nextTask_.fetch_add(1, std::memory_order_relaxed))
                invoke(job, task);
            lock.lock();
            assert(workingThreads_ > 0);
            if (--workingThreads_ == 0) finished_.notify_one();
        }
    }

    /// @brief Measure fixed host schedule storage before allocation.
    /// @details The order equals the constructor's aligned suballocations.
    /// @param capacity Positive schedule capacity.
    /// @return Exact payload layout, before pool-block rounding.
    GpuHostLayout Phase2TaskProjectionArena::hostLayout(Phase2TaskProjectionCapacity capacity) {
        GpuHostLayout layout;
        layout.add<DevicePhase2Task>(capacity.tasks);
        layout.add<DeviceRequestBatch>(capacity.batches);
        layout.add<DeviceMandatoryTerm>(capacity.terms);
        layout.add<DeviceExpressionStump>(capacity.stumps);
        return layout;
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
        hostStorage_.reserve(hostLayout(capacity).bytes);
        tasks.bindStorage(hostStorage_);
        tasks.reserve(capacity.tasks);
        batches.bindStorage(hostStorage_);
        batches.reserve(capacity.batches);
        terms.bindStorage(hostStorage_);
        terms.reserve(capacity.terms);
        stumps.bindStorage(hostStorage_);
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

    /// @brief Measure fixed host schedule storage before allocation.
    /// @details The order equals the constructor's aligned suballocations.
    /// @param capacity Positive schedule capacity.
    /// @return Exact payload layout, before pool-block rounding.
    GpuHostLayout Phase2FilterScheduleArena::hostLayout(Phase2FilterScheduleCapacity capacity) {
        GpuHostLayout layout;
        assert(capacity.calls > 0 && capacity.calls <= UINT32_MAX / 2);
        uint32_t slots = 1;
        while (slots < capacity.calls * 2u) {
            assert(slots <= UINT32_MAX / 2);
            slots <<= 1u;
        }
        layout.add<DevicePhase2FilterCall>(capacity.calls);
        layout.add<DevicePhase2FilterCall>(capacity.calls);
        layout.add<uint32_t>(capacity.calls);
        layout.add<uint32_t>(capacity.calls);
        layout.add<int32_t>(slots);
        return layout;
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
        assert(capacity.maximumExaminedRowsPerCall
            <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max()));
        assert(capacity.calls <= std::numeric_limits<uint32_t>::max() / 2u);
        uint32_t slotCapacity = 1;
        while (slotCapacity < capacity.calls * 2u) {
            assert(slotCapacity <= std::numeric_limits<uint32_t>::max() / 2u);
            slotCapacity <<= 1u;
        }
        hostStorage_.reserve(hostLayout(capacity).bytes);
        calls.bindStorage(hostStorage_);
        calls.reserve(capacity.calls);
        classes.bindStorage(hostStorage_);
        classes.reserve(capacity.calls);
        callClassIndices.bindStorage(hostStorage_);
        callClassIndices.reserve(capacity.calls);
        classMultiplicities.bindStorage(hostStorage_);
        classMultiplicities.reserve(capacity.calls);
        classSlots.bindStorage(hostStorage_);
        classSlots.reserve(slotCapacity);
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
    /// slice, interns the exact six-field descriptor with fixed open addressing,
    /// and appends its class index beside the processor-order call. No statement
    /// or map bytes are copied because CUDA reads the resident projection.
    ///
    /// @param logicalBlockIndex Resident logical-block projection index.
    /// @param memory Hash-memory registry supplying whole and subkey maps.
    /// @param maximumIterationNumberVariable Inclusive iteration ceiling.
    /// @param alsoAcceptFullKeys Bivalent processor whole-key widening flag.
    /// @param statementCount Full resident statement slice examined by the call.
    /// @param maximumFilteredStatements Positive eligible-row limit fitting the position codes.
    /// @return Pointer-free descriptor also appended to `calls`.
    /// @invariant Call order exactly matches the processor schedule; class order
    ///            is first occurrence; neither row total exceeds fixed ceilings.
    DevicePhase2FilterCall Phase2FilterScheduleArena::appendCall(
        uint32_t logicalBlockIndex,
        DeviceHashMemoryKind memory,
        int32_t maximumIterationNumberVariable,
        uint32_t alsoAcceptFullKeys,
        uint32_t statementCount,
        uint32_t maximumFilteredStatements) {
        assert(maximumFilteredStatements > 0
            && maximumFilteredStatements <= ExecutionParameters::MAX_FILTERED_POSITION_CODE);
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
        call.maximumFilteredStatements = maximumFilteredStatements;

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
        mix(call.maximumFilteredStatements);
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
                && candidate.statementCount == call.statementCount
                && candidate.maximumFilteredStatements == call.maximumFilteredStatements) {
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

    /// @brief Measure fixed host schedule storage before allocation.
    /// @details The order equals the constructor's aligned suballocations.
    /// @param capacity Positive schedule capacity.
    /// @return Exact payload layout, before pool-block rounding.
    GpuHostLayout Phase2GrowthScheduleArena::hostLayout(uint32_t capacity) {
        GpuHostLayout layout;
        layout.add<DevicePhase2GrowthCall>(capacity);
        return layout;
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
        hostStorage_.reserve(hostLayout(capacity).bytes);
        calls.bindStorage(hostStorage_);
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
