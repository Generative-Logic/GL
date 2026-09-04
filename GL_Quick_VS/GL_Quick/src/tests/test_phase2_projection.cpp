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

/// @file
/// @brief Direct twins for the fixed-capacity Phase 2 semantic projection.
///
/// @details
/// Tests compare statements, names, ranks, rule strings, all map and run shapes,
/// the remaining-argument reverse index, mandatory keys, metadata, cross-block
/// offsets, and allocation reuse. The CUDA twin uploads non-empty representatives
/// of all 24 columns, compares one device-computed checksum over every prefix, and
/// probes name, byte-key, reverse, and all eleven plain-data lookup views on-device.
/// The global filter/sort twin allocates the measured FTA ceilings and compares
/// every CUDA-retained index against processor filtering and stable name ordering.

#include "test_harness.hpp"

#include "../gpu/phase2_projection.hpp"
#ifdef GL_CUDA
#include "../gpu/phase2_cuda.hpp"
#include "../gpu/phase2_sealing.hpp"
#endif
#include "../memory.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <set>
#include <string>
#include <vector>

namespace {

gl::IntEncodedExpr projectionStatement(
    gl::NameId nameId,
    gl::NameId originalId,
    gl::NameId validityId,
    gl::NameId argumentId) {
    gl::IntEncodedExpr statement{};
    statement.nameId = nameId;
    statement.arity = 1;
    statement.maxIteration = -1;
    statement.originalId = originalId;
    statement.validityId = validityId;
    statement.argId[0] = argumentId;
    statement.argIteration[0] = -1;
    statement.argFullId[0] = argumentId;
    return statement;
}

std::string projectedName(
    const gl::gpu::Phase2ProjectionArena& arena,
    const gl::gpu::DeviceLogicalBlockProjection& block,
    gl::NameId id) {
    const gl::gpu::DeviceNameRecord& record =
        arena.nameRecords[block.nameRecordOffset + static_cast<uint32_t>(id)];
    return std::string(arena.nameBytes.data() + record.byteOffset,
                       static_cast<std::size_t>(record.byteLength));
}

uint64_t projectionChecksum(const gl::gpu::Phase2ProjectionArena& arena) {
    uint64_t value = 14695981039346656037ull;
    constexpr uint64_t prime = 1099511628211ull;
    const unsigned char* columns[24] = {
        reinterpret_cast<const unsigned char*>(arena.logicalBlocks.data()),
        reinterpret_cast<const unsigned char*>(arena.statements.data()),
        reinterpret_cast<const unsigned char*>(arena.nameRecords.data()),
        reinterpret_cast<const unsigned char*>(arena.nameBytes.data()),
        reinterpret_cast<const unsigned char*>(arena.nameSlots.data()),
        reinterpret_cast<const unsigned char*>(arena.ruleStringRecords.data()),
        reinterpret_cast<const unsigned char*>(arena.ruleStringBytes.data()),
        reinterpret_cast<const unsigned char*>(arena.byteMapViews.data()),
        reinterpret_cast<const unsigned char*>(arena.byteMapEntries.data()),
        reinterpret_cast<const unsigned char*>(arena.byteMapSlots.data()),
        reinterpret_cast<const unsigned char*>(arena.byteKeyBytes.data()),
        reinterpret_cast<const unsigned char*>(arena.blobRecords.data()),
        reinterpret_cast<const unsigned char*>(arena.blobBytes.data()),
        reinterpret_cast<const unsigned char*>(arena.reverseMapViews.data()),
        reinterpret_cast<const unsigned char*>(arena.reverseMapEntries.data()),
        reinterpret_cast<const unsigned char*>(arena.reverseMapSlots.data()),
        reinterpret_cast<const unsigned char*>(arena.reverseKeyBytes.data()),
        reinterpret_cast<const unsigned char*>(arena.reverseOwners.data()),
        reinterpret_cast<const unsigned char*>(arena.podMapViews.data()),
        reinterpret_cast<const unsigned char*>(arena.podMapEntries.data()),
        reinterpret_cast<const unsigned char*>(arena.podMapSlots.data()),
        reinterpret_cast<const unsigned char*>(arena.podRunValues.data()),
        reinterpret_cast<const unsigned char*>(arena.mandatoryStatementKeys.data()),
        reinterpret_cast<const unsigned char*>(arena.metadataBytes.data())
    };
    const std::size_t sizes[24] = {
        arena.logicalBlocks.size()
            * sizeof(gl::gpu::DeviceLogicalBlockProjection),
        arena.statements.size() * sizeof(gl::IntEncodedExpr),
        arena.nameRecords.size() * sizeof(gl::gpu::DeviceNameRecord),
        arena.nameBytes.size(),
        arena.nameSlots.size() * sizeof(int32_t),
        arena.ruleStringRecords.size()
            * sizeof(gl::gpu::DeviceRuleStringRecord),
        arena.ruleStringBytes.size(),
        arena.byteMapViews.size() * sizeof(gl::gpu::DeviceByteMapView),
        arena.byteMapEntries.size() * sizeof(gl::gpu::DeviceByteMapEntry),
        arena.byteMapSlots.size() * sizeof(int32_t),
        arena.byteKeyBytes.size(),
        arena.blobRecords.size() * sizeof(gl::gpu::DeviceBlobRecord),
        arena.blobBytes.size(),
        arena.reverseMapViews.size() * sizeof(gl::gpu::DeviceReverseMapView),
        arena.reverseMapEntries.size() * sizeof(gl::gpu::DeviceReverseMapEntry),
        arena.reverseMapSlots.size() * sizeof(int32_t),
        arena.reverseKeyBytes.size(),
        arena.reverseOwners.size() * sizeof(int32_t),
        arena.podMapViews.size() * sizeof(gl::gpu::DevicePodMapView),
        arena.podMapEntries.size() * sizeof(gl::gpu::DevicePodMapEntry),
        arena.podMapSlots.size() * sizeof(int32_t),
        arena.podRunValues.size() * sizeof(int32_t),
        arena.mandatoryStatementKeys.size() * sizeof(int64_t),
        arena.metadataBytes.size()
    };
    for (int column = 0; column < 24; ++column) {
        for (std::size_t index = 0; index < sizes[column]; ++index) {
            value ^= static_cast<uint64_t>(columns[column][index]);
            value *= prime;
        }
    }
    return value;
}

uint64_t taskProjectionChecksum(
    const gl::gpu::Phase2TaskProjectionArena& arena) {
    uint64_t value = 14695981039346656037ull;
    constexpr uint64_t prime = 1099511628211ull;
    const unsigned char* columns[4] = {
        reinterpret_cast<const unsigned char*>(arena.tasks.data()),
        reinterpret_cast<const unsigned char*>(arena.batches.data()),
        reinterpret_cast<const unsigned char*>(arena.terms.data()),
        reinterpret_cast<const unsigned char*>(arena.stumps.data())
    };
    const std::size_t sizes[4] = {
        arena.tasks.size() * sizeof(gl::gpu::DevicePhase2Task),
        arena.batches.size() * sizeof(gl::gpu::DeviceRequestBatch),
        arena.terms.size() * sizeof(gl::gpu::DeviceMandatoryTerm),
        arena.stumps.size() * sizeof(gl::gpu::DeviceExpressionStump)
    };
    for (int column = 0; column < 4; ++column) {
        for (std::size_t index = 0; index < sizes[column]; ++index) {
            value ^= static_cast<uint64_t>(columns[column][index]);
            value *= prime;
        }
    }
    return value;
}

gl::gpu::Phase2ProjectionCapacity projectionCapacity(
    uint32_t logicalBlocks,
    uint32_t statements,
    uint32_t nameRecords,
    uint32_t nameBytes) {
    gl::gpu::Phase2ProjectionCapacity capacity{};
    capacity.logicalBlocks = logicalBlocks;
    capacity.statements = statements;
    capacity.nameRecords = nameRecords;
    capacity.nameBytes = nameBytes;
    capacity.nameSlots = 1024;
    capacity.ruleStringRecords = 256;
    capacity.ruleStringBytes = 16384;
    capacity.byteMapViews = logicalBlocks * 10;
    capacity.byteMapEntries = 1024;
    capacity.byteMapSlots = 4096;
    capacity.byteKeyBytes = 65536;
    capacity.blobRecords = 1024;
    capacity.blobBytes = 65536;
    capacity.reverseMapViews = logicalBlocks;
    capacity.reverseMapEntries = 1024;
    capacity.reverseMapSlots = 4096;
    capacity.reverseKeyBytes = 65536;
    capacity.reverseOwners = 1024;
    capacity.podMapViews = logicalBlocks * 11;
    capacity.podMapEntries = 1024;
    capacity.podMapSlots = 4096;
    capacity.podRunValues = 1024;
    capacity.mandatoryStatementKeys = 1024;
    capacity.metadataBytes = 4096;
    return capacity;
}

}  // namespace

TEST(phase2_projection, backend_defaults_to_processor_and_accepts_explicit_cuda) {
    // The product default is the processor route (`use_gpu` false, no
    // shipped config pins a batch); it keeps its SSD permission and pages.
    gl::ExpressionAnalyzer defaultAnalyzer("FTA");
    ASSERT_FALSE(defaultAnalyzer.parameters.use_gpu);
    ASSERT_TRUE(defaultAnalyzer.parameters.allow_ssd_deload);
    ASSERT_EQ(defaultAnalyzer.phase2Backend, gl::Phase2Backend::cpu);
#ifdef GL_CUDA
    // `main.py --GPU` arrives as the explicit CUDA override; the
    // backend-derived SSD policy makes a CUDA batch resident-only. A build
    // without CUDA support refuses the selection at construction instead.
    gl::ExpressionAnalyzer cudaAnalyzer("FTA", gl::Phase2Backend::cuda);
    ASSERT_EQ(cudaAnalyzer.phase2Backend, gl::Phase2Backend::cuda);
    ASSERT_FALSE(cudaAnalyzer.parameters.allow_ssd_deload);
#endif
    // The explicit processor override is accepted as well.
    gl::ExpressionAnalyzer cpuAnalyzer("FTA", gl::Phase2Backend::cpu);
    ASSERT_EQ(cpuAnalyzer.phase2Backend, gl::Phase2Backend::cpu);
    ASSERT_TRUE(cpuAnalyzer.parameters.allow_ssd_deload);
}

TEST(phase2_projection, physical_block_and_task_ceilings_are_independent) {
    const gl::gpu::Phase2ProjectionCapacity shortcut =
        gl::gpu::phase2ProjectionCapacityFor(
            gl::gpu::Phase2ProjectionProfile::ftaShortcut);
    const gl::gpu::Phase2ProjectionCapacity fullRun =
        gl::gpu::phase2ProjectionCapacityFor(
            gl::gpu::Phase2ProjectionProfile::fullRun);
    ASSERT_EQ(shortcut.logicalBlocks,
        gl::gpu::kMaxProjectedBlocksPerChunk);
    ASSERT_EQ(fullRun.logicalBlocks,
        gl::gpu::kMaxProjectedBlocksPerChunk);
    ASSERT_EQ(gl::gpu::kMaxProjectedBlocksPerChunk, 512u);
    ASSERT_EQ(gl::gpu::kMaxPhase2TasksPerChunk, 1024u);
    ASSERT_EQ(shortcut.nameRecords, 1048576u);
    ASSERT_EQ(shortcut.byteKeyBytes, 134217728u);
    ASSERT_EQ(fullRun.nameRecords, 6540000u);
    ASSERT_EQ(fullRun.byteKeyBytes, 2000000000u);
    ASSERT_EQ(gl::gpu::kPhase2ProjectionConstructionShardCount, 12u);
    ASSERT_EQ(gl::gpu::kPhase2ProjectionConstructionShardAverageShares, 2u);
    ASSERT_EQ(gl::gpu::kPhase2ProjectionConstructionShardShareDivisor, 6u);
    ASSERT_EQ(gl::gpu::kPhase2TaskProjectionCapacity.tasks, 1024u);
    ASSERT_EQ(gl::gpu::kPhase2TaskProjectionCapacity.batches, 4096u);
    ASSERT_EQ(gl::gpu::kPhase2TaskProjectionCapacity.terms, 4096u);
    ASSERT_EQ(gl::gpu::kPhase2TaskProjectionCapacity.stumps, 16384u);
    ASSERT_EQ(gl::gpu::kPhase2FilterScheduleCapacity.calls, 4096u);
    ASSERT_EQ(gl::gpu::kPhase2FilterScheduleCapacity.examinedRows, 16777216u);
    ASSERT_EQ(gl::gpu::kPhase2FilterScheduleCapacity.retainedRows, 1048576u);
    ASSERT_EQ(gl::gpu::kPhase2FilterScheduleCapacity.maximumExaminedRowsPerCall,
        262144u);
    ASSERT_EQ(gl::gpu::kFtaShortcutPhase2GrowthCapacity.calls, 4096u);
    ASSERT_EQ(gl::gpu::kFtaShortcutPhase2GrowthCapacity.rawRequests, 262144u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.calls, 4096u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.rawRequests, 524288u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.frontierRecords, 4194304u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.prefixPayloadValues,
        gl::gpu::kFtaShortcutPhase2GrowthCapacity.prefixPayloadValues);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.prefixVariableValues,
        gl::gpu::kFtaShortcutPhase2GrowthCapacity.prefixVariableValues);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.prefixSecondaryValues,
        gl::gpu::kFtaShortcutPhase2GrowthCapacity.prefixSecondaryValues);
#ifdef GL_CUDA
    ASSERT_EQ(gl::gpu::kFtaShortcutPhase2GrowthCapacity.frontierRecords,
        gl::gpu::kFtaShortcutPhase2OrderingCapacity.events);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.acceptedEvents,
        gl::gpu::kFullRunPhase2OrderingCapacity.events);
    ASSERT_EQ(gl::gpu::kFtaShortcutPhase2OrderingCapacity.requests,
        gl::gpu::kFtaShortcutPhase2EvaluationCapacity.requests);
    ASSERT_EQ(gl::gpu::kFullRunPhase2OrderingCapacity.requests,
        gl::gpu::kFullRunPhase2EvaluationCapacity.requests);
    ASSERT_EQ(gl::gpu::kFtaShortcutPhase2EvaluationCapacity.logicalBlocks,
        gl::gpu::kMaxProjectedBlocksPerChunk);
    ASSERT_EQ(gl::gpu::kFullRunPhase2EvaluationCapacity.logicalBlocks,
        gl::gpu::kMaxProjectedBlocksPerChunk);
    ASSERT_EQ(gl::gpu::kFtaShortcutPhase2EvaluationCapacity.markerKeys,
        131072u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2EvaluationCapacity.markerKeys, 524288u);
#endif
}

TEST(phase2_projection, rows_names_parents_and_ranks_are_lossless) {
    gl::ExpressionAnalyzer analyzer("Peano");
    gl::Memory body;
    const gl::NameId mainId = body.nameMap.encode("main");
    const gl::NameId zetaId = body.nameMap.encode("zeta");
    const gl::NameId alphaId = body.nameMap.encode("alpha");
    const gl::NameId childId = body.nameMap.encodePush(mainId, "child");
    const gl::NameId originalA = body.nameMap.encode("(zeta[alpha])");
    const gl::NameId originalB = body.nameMap.encode("(alpha[zeta])");
    body.nameMap.encode("prefix");
    body.nameMap.encode("prefix0");
    body.nameMap.encode("prefix00");
    body.nameMap.encode("prefix01");
    body.nameMap.encode("prefix1");

    const gl::IntEncodedExpr first =
        projectionStatement(zetaId, originalA, mainId, alphaId);
    const gl::IntEncodedExpr second =
        projectionStatement(alphaId, originalB, childId, zetaId);
    body.intEncodedStatements.push_back(first);
    body.intEncodedStatements.push_back(second);

    uint32_t byteCount = 0;
    for (gl::NameId id = 1; id <= body.nameMap.nameCount(); ++id)
        byteCount += static_cast<uint32_t>(body.nameMap.decodeView(id).len);

    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(
        1,
        2,
        static_cast<uint32_t>(body.nameMap.nameCount() + 1),
        byteCount));
    const gl::gpu::DeviceLogicalBlockProjection projected =
        arena.appendLogicalBlock(body, analyzer);

    ASSERT_EQ(projected.statementOffset, 0u);
    ASSERT_EQ(projected.statementCount, 2u);
    ASSERT_EQ(projected.nameRecordOffset, 0u);
    ASSERT_EQ(projected.nameRecordCount,
              static_cast<uint32_t>(body.nameMap.nameCount() + 1));
    ASSERT_EQ(projected.nameByteCount, byteCount);
    ASSERT_TRUE(std::memcmp(&arena.statements[0], &first, sizeof(first)) == 0);
    ASSERT_TRUE(std::memcmp(&arena.statements[1], &second, sizeof(second)) == 0);

    for (gl::NameId id = 1; id <= body.nameMap.nameCount(); ++id) {
        ASSERT_EQ(projectedName(arena, projected, id), body.nameMap.decode(id));
        ASSERT_EQ(arena.nameRecords[static_cast<uint32_t>(id)].parentId,
                  body.nameMap.parentOf(id));
    }
    ASSERT_EQ(arena.nameRecords[static_cast<uint32_t>(childId)].parentId, mainId);
    ASSERT_LT(arena.nameRecords[static_cast<uint32_t>(alphaId)].decodedLexRank,
              arena.nameRecords[static_cast<uint32_t>(zetaId)].decodedLexRank);

    std::vector<gl::NameId> expectedOrder;
    for (gl::NameId id = 1; id <= body.nameMap.nameCount(); ++id)
        expectedOrder.push_back(id);
    std::sort(expectedOrder.begin(), expectedOrder.end(),
        [&body](gl::NameId left, gl::NameId right) {
            return gl::compareSpans(body.nameMap.decodeView(left),
                                    body.nameMap.decodeView(right)) < 0;
        });
    for (uint32_t rank = 0; rank < expectedOrder.size(); ++rank) {
        const gl::NameId id = expectedOrder[rank];
        ASSERT_EQ(arena.nameRecords[static_cast<uint32_t>(id)].decodedLexRank,
                  rank);
    }
}

TEST(phase2_projection, append_offsets_and_clear_reuse_fixed_storage) {
    gl::ExpressionAnalyzer analyzer("Peano");
    gl::Memory firstBody;
    const gl::NameId firstMain = firstBody.nameMap.encode("main");
    const gl::NameId firstName = firstBody.nameMap.encode("beta");
    firstBody.intEncodedStatements.push_back(
        projectionStatement(firstName, firstName, firstMain, firstName));

    gl::Memory secondBody;
    const gl::NameId secondMain = secondBody.nameMap.encode("main");
    const gl::NameId secondName = secondBody.nameMap.encode("gamma");
    secondBody.intEncodedStatements.push_back(
        projectionStatement(secondName, secondName, secondMain, secondName));

    uint32_t bytes = 0;
    for (gl::NameId id = 1; id <= firstBody.nameMap.nameCount(); ++id)
        bytes += static_cast<uint32_t>(firstBody.nameMap.decodeView(id).len);
    for (gl::NameId id = 1; id <= secondBody.nameMap.nameCount(); ++id)
        bytes += static_cast<uint32_t>(secondBody.nameMap.decodeView(id).len);

    const uint32_t records =
        static_cast<uint32_t>(firstBody.nameMap.nameCount()
            + secondBody.nameMap.nameCount() + 2);
    gl::gpu::Phase2ProjectionArena arena(
        projectionCapacity(2, 2, records, bytes));
    const gl::IntEncodedExpr* statementAllocation = arena.statements.data();
    const gl::gpu::DeviceNameRecord* nameAllocation = arena.nameRecords.data();
    const char* byteAllocation = arena.nameBytes.data();

    const auto firstProjection = arena.appendLogicalBlock(firstBody, analyzer);
    const auto secondProjection = arena.appendLogicalBlock(secondBody, analyzer);
    ASSERT_EQ(secondProjection.statementOffset,
              firstProjection.statementOffset + firstProjection.statementCount);
    ASSERT_EQ(secondProjection.nameRecordOffset,
              firstProjection.nameRecordOffset + firstProjection.nameRecordCount);
    ASSERT_EQ(secondProjection.nameByteOffset,
              firstProjection.nameByteOffset + firstProjection.nameByteCount);
    ASSERT_EQ(arena.statements.size(), 2u);
    ASSERT_EQ(arena.nameRecords.size(), static_cast<std::size_t>(records));
    ASSERT_EQ(arena.nameBytes.size(), static_cast<std::size_t>(bytes));
    ASSERT_EQ(arena.timing.logicalBlocks, 2u);

    arena.clear();
    ASSERT_EQ(arena.statements.data(), statementAllocation);
    ASSERT_EQ(arena.nameRecords.data(), nameAllocation);
    ASSERT_EQ(arena.nameBytes.data(), byteAllocation);
    ASSERT_TRUE(arena.logicalBlocks.empty());
    ASSERT_TRUE(arena.statements.empty());
    ASSERT_TRUE(arena.nameRecords.empty());
    ASSERT_TRUE(arena.nameBytes.empty());
    ASSERT_EQ(arena.timing.logicalBlocks, 0u);
    ASSERT_EQ(arena.timing.preflightNanoseconds, 0u);
    ASSERT_EQ(arena.timing.statementNanoseconds, 0u);
    ASSERT_EQ(arena.timing.nameNanoseconds, 0u);
    ASSERT_EQ(arena.timing.nameRecordNanoseconds, 0u);
    ASSERT_EQ(arena.timing.nameSortNanoseconds, 0u);
    ASSERT_EQ(arena.timing.nameRankNanoseconds, 0u);
    ASSERT_EQ(arena.timing.nameSlotNanoseconds, 0u);
    ASSERT_EQ(arena.timing.ruleStringNanoseconds, 0u);
    ASSERT_EQ(arena.timing.byteMapNanoseconds, 0u);
    ASSERT_EQ(arena.timing.reverseMapNanoseconds, 0u);
    ASSERT_EQ(arena.timing.podMapNanoseconds, 0u);
    ASSERT_EQ(arena.timing.finalNanoseconds, 0u);
    ASSERT_EQ(arena.timing.mergeNanoseconds, 0u);

    const auto reused = arena.appendLogicalBlock(firstBody, analyzer);
    ASSERT_EQ(reused.statementOffset, 0u);
    ASSERT_EQ(reused.nameRecordOffset, 0u);
    ASSERT_EQ(reused.nameByteOffset, 0u);
    ASSERT_EQ(arena.timing.logicalBlocks, 1u);
}

TEST(phase2_projection, identical_read_only_string_columns_share_exactly) {
    gl::ExpressionAnalyzer analyzer("Peano");
    gl::Memory firstBody;
    const gl::NameId firstMain = firstBody.nameMap.encode("main");
    firstBody.nameMap.encodePush(firstMain, "child");
    firstBody.ruleInterner.encode("rule_a");

    gl::Memory identicalBody;
    const gl::NameId identicalMain = identicalBody.nameMap.encode("main");
    identicalBody.nameMap.encodePush(identicalMain, "child");
    identicalBody.ruleInterner.encode("rule_a");

    gl::Memory differentParentBody;
    differentParentBody.nameMap.encode("main");
    differentParentBody.nameMap.encode("child");
    differentParentBody.ruleInterner.encode("rule_a");

    gl::Memory differentRuleBody;
    const gl::NameId differentRuleMain =
        differentRuleBody.nameMap.encode("main");
    differentRuleBody.nameMap.encodePush(differentRuleMain, "child");
    differentRuleBody.ruleInterner.encode("rule_b");

    gl::gpu::Phase2ProjectionArena arena(
        projectionCapacity(4, 4, 16, 128));
    const auto first = arena.appendLogicalBlock(firstBody, analyzer);
    const std::size_t firstNameRecords = arena.nameRecords.size();
    const std::size_t firstNameBytes = arena.nameBytes.size();
    const std::size_t firstNameSlots = arena.nameSlots.size();
    const std::size_t firstRuleRecords = arena.ruleStringRecords.size();
    const std::size_t firstRuleBytes = arena.ruleStringBytes.size();

    const auto identical =
        arena.appendLogicalBlock(identicalBody, analyzer);
    ASSERT_EQ(identical.nameRecordOffset, first.nameRecordOffset);
    ASSERT_EQ(identical.nameByteOffset, first.nameByteOffset);
    ASSERT_EQ(identical.nameSlotOffset, first.nameSlotOffset);
    ASSERT_EQ(identical.ruleStringRecordOffset,
              first.ruleStringRecordOffset);
    ASSERT_EQ(identical.ruleStringByteOffset, first.ruleStringByteOffset);
    ASSERT_EQ(arena.nameRecords.size(), firstNameRecords);
    ASSERT_EQ(arena.nameBytes.size(), firstNameBytes);
    ASSERT_EQ(arena.nameSlots.size(), firstNameSlots);
    ASSERT_EQ(arena.ruleStringRecords.size(), firstRuleRecords);
    ASSERT_EQ(arena.ruleStringBytes.size(), firstRuleBytes);

    const auto differentParent =
        arena.appendLogicalBlock(differentParentBody, analyzer);
    ASSERT_NE(differentParent.nameRecordOffset, first.nameRecordOffset);
    ASSERT_EQ(differentParent.ruleStringRecordOffset,
              first.ruleStringRecordOffset);

    const auto differentRule =
        arena.appendLogicalBlock(differentRuleBody, analyzer);
    ASSERT_EQ(differentRule.nameRecordOffset, first.nameRecordOffset);
    ASSERT_NE(differentRule.ruleStringRecordOffset,
              first.ruleStringRecordOffset);
}

TEST(phase2_projection, shard_merge_rebases_every_semantic_column_exactly) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.compiledExpressions["theta"] = gl::LogicalEntity(
        "atomic", {}, "theta", 1);

    gl::Memory prefixBody;
    const gl::NameId prefixMain = prefixBody.nameMap.encode("main");
    const gl::NameId prefixName = prefixBody.nameMap.encode("prefix");
    prefixBody.setExprKey("prefix_block");
    prefixBody.intEncodedStatements.push_back(projectionStatement(
        prefixName, prefixName, prefixMain, prefixName));

    gl::Memory richBody;
    const gl::NameId mainId = richBody.nameMap.encode("main");
    const gl::NameId nameId = richBody.nameMap.encode("theta");
    richBody.setExprKey("rich_block");
    const gl::IntEncodedExpr statement = projectionStatement(
        nameId, nameId, mainId, nameId);
    richBody.intEncodedStatements.push_back(statement);
    richBody.intLocalEncodedStatements.push_back(statement);
    richBody.intLocalEncodedStatementsDelta.push_back(statement);
    richBody.intExternalStatements.push_back(statement);

    const int32_t ruleHeadId = richBody.ruleInterner.encode("(theta[1])");
    const int32_t ruleSourceId = richBody.ruleInterner.encode(
        "(>[theta,(theta[1])])");
    const gl::NormKey normKey{ 1, { nameId, mainId } };
    gl::OwnerSet owner;
    owner.hasLooseOwner = true;
    const gl::RuleOwner packedOwner = gl::packRuleOwner(
        ruleSourceId, mainId);
    owner.owners.emplace_back(packedOwner, -1);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::LocalMemoryValue value;
    value.valueId = ruleHeadId;
    value.originalImplicationId = ruleSourceId;
    value.validityId = mainId;
    value.levels.insert(3);
    value.keyIds.push_back(ruleHeadId);
    value.remainingArgIds.push_back(ruleHeadId);
    const gl::Int16SetKey remainingKey{ { nameId } };
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        richBody.overallHashMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    richBody.overallHashMemory.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    richBody.overallHashMemory.encodedMap.assignRun(
        normKey, std::vector<gl::LocalMemoryValue>{ value });
    richBody.overallHashMemory.remainingArgsNormalizedEncodedMap.assignRun(
        remainingKey, std::vector<gl::NormKey>{ normKey });
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        richBody.localHashMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    richBody.localHashMemory.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        richBody.localHashMemoryDelta.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    richBody.localHashMemoryDelta.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        richBody.workingMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    richBody.workingMemory.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    richBody.overallHashMemory.productsOfRecursionIds.mint(nameId);
    const gl::StatementKey statementKey{ nameId, mainId };
    const int64_t packedStatementKey = gl::packStatementKey(nameId, mainId);
    richBody.intKnownStatements.upsert(
        statementKey, gl::StatementFlags{ true, true });
    richBody.intLocalEncodedStatementsSet.mint(packedStatementKey);
    const int levels[1] = { 4 };
    richBody.intStatementLevelsMap.assignSet(statementKey, levels, 1);
    richBody.intValidityNamesToFilter.mint(mainId);
    richBody.frozenOrBranches.mint(nameId);
    richBody.intToBeProved.assignSet(statementKey, levels, 1);
    richBody.canBeSentIds.mint(nameId);
    richBody.canBeSentMarkerIds.mint(nameId);

    const gl::gpu::Phase2ProjectionCapacity capacity = projectionCapacity(
        2, 2, 16, 512);
    gl::gpu::Phase2ProjectionArena serial(capacity);
    serial.appendLogicalBlock(prefixBody, analyzer);
    serial.appendLogicalBlock(richBody, analyzer);

    gl::gpu::Phase2ProjectionArena prefixShard(capacity);
    gl::gpu::Phase2ProjectionArena richShard(capacity);
    prefixShard.appendLogicalBlock(prefixBody, analyzer);
    richShard.appendLogicalBlock(richBody, analyzer);
    ASSERT_TRUE(!richShard.byteMapEntries.empty());
    ASSERT_TRUE(!richShard.blobRecords.empty());
    ASSERT_TRUE(!richShard.reverseMapEntries.empty());
    ASSERT_TRUE(!richShard.podMapEntries.empty());
    ASSERT_TRUE(!richShard.podRunValues.empty());

    gl::gpu::Phase2ProjectionArena merged(capacity);
    ASSERT_EQ(merged.appendShard(prefixShard), 0u);
    ASSERT_EQ(merged.appendShard(richShard), 1u);
    ASSERT_EQ(merged.timing.logicalBlocks, 2u);
    ASSERT_EQ(merged.logicalBlocks.size(), serial.logicalBlocks.size());
    ASSERT_EQ(merged.statements.size(), serial.statements.size());
    ASSERT_EQ(merged.nameRecords.size(), serial.nameRecords.size());
    ASSERT_EQ(merged.byteMapEntries.size(), serial.byteMapEntries.size());
    ASSERT_EQ(merged.reverseMapEntries.size(), serial.reverseMapEntries.size());
    ASSERT_EQ(merged.podMapEntries.size(), serial.podMapEntries.size());
    ASSERT_EQ(projectionChecksum(merged), projectionChecksum(serial));
}

TEST(phase2_projection, complete_usage_counts_every_semantic_column) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.incubator_mode = true;
    analyzer.parameters.ban_disintegration = false;
    analyzer.parameters.compressor_mode = true;
    analyzer.parameters.standardMaxSecondaryNumber = 13;
    analyzer.compiledExpressions["theta"] = gl::LogicalEntity(
        "atomic", {}, "theta", 1);
    gl::Memory body;
    body.level = 7;
    const gl::NameId mainId = body.nameMap.encode("main");
    const gl::NameId nameId = body.nameMap.encode("theta");
    body.setExprKey("gpu_test");
    const gl::IntEncodedExpr statement =
        projectionStatement(nameId, nameId, mainId, nameId);
    body.intEncodedStatements.push_back(statement);
    body.intLocalEncodedStatements.push_back(statement);
    body.intLocalEncodedStatementsDelta.push_back(statement);
    body.intExternalStatements.push_back(statement);

    const int32_t ruleHeadId = body.ruleInterner.encode("(theta[1])");
    const int32_t ruleSourceId =
        body.ruleInterner.encode("(>[theta,(theta[1])])");
    const gl::NormKey normKey{ 1, { nameId, mainId } };
    gl::OwnerSet owner;
    owner.hasLooseOwner = true;
    const gl::RuleOwner packedOwner = gl::packRuleOwner(
        ruleSourceId, mainId);
    owner.owners.emplace_back(packedOwner, -1);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::LocalMemoryValue value;
    value.valueId = ruleHeadId;
    value.originalImplicationId = ruleSourceId;
    value.validityId = mainId;
    value.levels.insert(3);
    value.keyIds.push_back(ruleHeadId);
    value.remainingArgIds.push_back(ruleHeadId);
    const gl::Int16SetKey remainingKey{ { nameId } };

    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.overallHashMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    body.overallHashMemory.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    body.overallHashMemory.encodedMap.assignRun(
        normKey, std::vector<gl::LocalMemoryValue>{ value });
    body.overallHashMemory.remainingArgsNormalizedEncodedMap.assignRun(
        remainingKey, std::vector<gl::NormKey>{ normKey });
    body.overallHashMemory.productsOfRecursionIds.mint(nameId);
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.localHashMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.localHashMemoryDelta.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.workingMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);

    const gl::StatementKey statementKey{ nameId, mainId };
    const int64_t packedStatementKey = gl::packStatementKey(nameId, mainId);
    body.intKnownStatements.upsert(
        statementKey, gl::StatementFlags{ true, false });
    body.intLocalEncodedStatementsSet.mint(packedStatementKey);
    const int levels[2] = { 2, 5 };
    body.intStatementLevelsMap.assignSet(statementKey, levels, 2);
    body.intValidityNamesToFilter.mint(mainId);
    body.frozenOrBranches.mint(nameId);
    const int goalAux[1] = { 7 };
    body.intToBeProved.assignSet(statementKey, goalAux, 1);
    body.canBeSentIds.mint(nameId);
    body.canBeSentMarkerIds.mint(nameId);

    const gl::gpu::Phase2ProjectionUsage usage =
        gl::gpu::measurePhase2ProjectionUsage(body, analyzer);
    ASSERT_EQ(usage.logicalBlocks, 1u);
    ASSERT_EQ(usage.statements, 1u);
    ASSERT_EQ(usage.nameSlots, 4u);
    ASSERT_EQ(usage.ruleStringRecords, 3u);
    ASSERT_EQ(usage.ruleStringBytes,
        static_cast<uint64_t>(body.ruleInterner.decodeView(ruleHeadId).len
            + body.ruleInterner.decodeView(ruleSourceId).len));
    ASSERT_EQ(usage.byteMapViews, 10u);
    ASSERT_EQ(usage.byteMapEntries, 7u);
    ASSERT_EQ(usage.byteMapSlots, 14u);
    const uint64_t normKeyBytes =
        static_cast<uint64_t>(gl::Codec<gl::NormKey>::encode(normKey).size());
    const uint64_t remainingKeyBytes =
        static_cast<uint64_t>(gl::Codec<gl::Int16SetKey>::encode(
            remainingKey).size());
    ASSERT_EQ(usage.byteKeyBytes, normKeyBytes * 6 + remainingKeyBytes);
    ASSERT_EQ(usage.blobRecords, 3u);
    ASSERT_EQ(usage.blobBytes,
        static_cast<uint64_t>(gl::Codec<gl::OwnerSet>::serialize(owner).size()
            + gl::Codec<gl::LocalMemoryValue>::serialize(value).size()
            + gl::Codec<gl::NormKey>::serialize(normKey).size()));
    ASSERT_EQ(usage.reverseMapViews, 1u);
    ASSERT_EQ(usage.reverseMapEntries, 1u);
    ASSERT_EQ(usage.reverseMapSlots, 2u);
    ASSERT_EQ(usage.reverseKeyBytes, normKeyBytes);
    ASSERT_EQ(usage.reverseOwners, 1u);
    ASSERT_EQ(usage.podMapViews, 11u);
    ASSERT_EQ(usage.podMapEntries, 11u);
    ASSERT_EQ(usage.podMapSlots, 22u);
    ASSERT_EQ(usage.podRunValues, 3u);
    ASSERT_EQ(usage.mandatoryStatementKeys, 3u);
    ASSERT_EQ(usage.metadataBytes,
        8u + static_cast<uint64_t>(analyzer.anchorInfo.name.size()));

    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(
        1, 1, static_cast<uint32_t>(body.nameMap.nameCount() + 1), 256));
    const gl::gpu::DeviceLogicalBlockProjection projected =
        arena.appendLogicalBlock(body, analyzer);
    ASSERT_EQ(projected.byteMapViewCount, 10u);
    ASSERT_EQ(projected.reverseMapViewCount, 1u);
    ASSERT_EQ(projected.podMapViewCount, 11u);
    ASSERT_EQ(projected.ruleStringRecordCount, 3u);
    ASSERT_EQ(projected.localKeyCount, 1u);
    ASSERT_EQ(projected.deltaKeyCount, 1u);
    ASSERT_EQ(projected.externalKeyCount, 1u);
    ASSERT_EQ(projected.metadataCount, 8u);
    ASSERT_TRUE(std::memcmp(
        arena.metadataBytes.data() + projected.metadataOffset,
        "gpu_test", 8) == 0);
    ASSERT_EQ(projected.anchorNameCount,
        static_cast<uint32_t>(analyzer.anchorInfo.name.size()));
    ASSERT_TRUE(std::memcmp(
        arena.metadataBytes.data() + projected.anchorNameOffset,
        analyzer.anchorInfo.name.data(), analyzer.anchorInfo.name.size()) == 0);
    ASSERT_EQ(projected.level, 7);
    ASSERT_EQ(projected.standardMaxSecondaryNumber, 13);
    ASSERT_EQ(projected.evaluationFlags,
        static_cast<uint32_t>(
            gl::gpu::deviceEvaluationIncubatorMode
            | gl::gpu::deviceEvaluationCompressorMode));
    ASSERT_EQ(arena.ruleStringRecords[
        projected.ruleStringRecordOffset
            + static_cast<uint32_t>(ruleHeadId)].compiledCategory,
        gl::gpu::DeviceCompiledCategory::atomic);
    ASSERT_EQ(arena.reverseMapViews[projected.reverseMapViewOffset].entryCount,
              1u);
    const gl::gpu::DeviceReverseMapEntry& reverseEntry =
        arena.reverseMapEntries[
            arena.reverseMapViews[projected.reverseMapViewOffset].entryOffset];
    ASSERT_EQ(reverseEntry.keyLength, normKeyBytes);
    ASSERT_EQ(reverseEntry.ownerCount, 1u);
    ASSERT_EQ(arena.reverseOwners[reverseEntry.ownerOffset], 1);
    ASSERT_TRUE(std::memcmp(
        arena.reverseKeyBytes.data() + reverseEntry.keyOffset,
        gl::Codec<gl::NormKey>::serialize(normKey).data(),
        static_cast<std::size_t>(normKeyBytes)) == 0);
    const gl::gpu::DevicePodMapView& knownView =
        arena.podMapViews[projected.podMapViewOffset + 1];
    ASSERT_EQ(knownView.kind,
              gl::gpu::DevicePodMapKind::knownStatements);
    ASSERT_EQ(knownView.entryCount, 1u);
    ASSERT_EQ(arena.podMapEntries[knownView.entryOffset].key,
              packedStatementKey);
    ASSERT_EQ(arena.podMapEntries[knownView.entryOffset].scalar, 1u);

    const uint32_t localOnly = 1u << static_cast<uint32_t>(
        gl::gpu::DeviceHashMemoryKind::local);
    const gl::gpu::Phase2ProjectionUsage selectedUsage =
        gl::gpu::measurePhase2ProjectionUsage(
            body, analyzer, localOnly);
    ASSERT_EQ(selectedUsage.byteMapViews, 10u);
    ASSERT_LT(selectedUsage.byteMapEntries, usage.byteMapEntries);
    ASSERT_LT(selectedUsage.byteMapSlots, usage.byteMapSlots);
    ASSERT_LT(selectedUsage.byteKeyBytes, usage.byteKeyBytes);

    gl::gpu::Phase2ProjectionArena selectedArena(projectionCapacity(
        1, 1, static_cast<uint32_t>(body.nameMap.nameCount() + 1), 256));
    const gl::gpu::DeviceLogicalBlockProjection selected =
        selectedArena.appendLogicalBlock(body, analyzer, localOnly);
    ASSERT_EQ(selected.byteMapViewCount, 10u);
    for (uint32_t ordinal = 0; ordinal < selected.byteMapViewCount;
         ++ordinal) {
        ASSERT_EQ(static_cast<uint32_t>(selectedArena.byteMapViews[
                      selected.byteMapViewOffset + ordinal].kind), ordinal);
    }
    ASSERT_EQ(selectedArena.byteMapViews[selected.byteMapViewOffset].entryCount,
        0u);
    ASSERT_EQ(selectedArena.byteMapViews[
        selected.byteMapViewOffset + 1].entryCount, 0u);
    ASSERT_EQ(selectedArena.byteMapViews[selected.byteMapViewOffset + 2].kind,
        gl::gpu::DeviceByteMapKind::localWholeKeys);
    ASSERT_EQ(selectedArena.byteMapViews[selected.byteMapViewOffset + 3].kind,
        gl::gpu::DeviceByteMapKind::localSubkeys);
    ASSERT_EQ(selectedArena.byteMapViews[selected.byteMapViewOffset + 8].kind,
        gl::gpu::DeviceByteMapKind::overallEncoded);
    ASSERT_EQ(selectedArena.byteMapViews[selected.byteMapViewOffset + 9].kind,
        gl::gpu::DeviceByteMapKind::overallRemainingArgs);
}

TEST(phase2_projection, task_inputs_are_pointer_free_ordered_and_reusable) {
    gl::Memory measuredBody;
    const gl::NameId measuredMain = measuredBody.nameMap.encode("main");
    const gl::NameId measuredName = measuredBody.nameMap.encode("task-shape");
    const gl::IntEncodedExpr measuredStatement = projectionStatement(
        measuredName, measuredName, measuredMain, measuredName);
    measuredBody.intLocalEncodedStatements.push_back(measuredStatement);
    measuredBody.intExternalStatements.push_back(measuredStatement);
    const gl::NormKey measuredKey{ 1, { measuredName } };
    gl::LocalMemoryValue measuredValue;
    measuredValue.valueId = measuredBody.ruleInterner.encode("task-rule");
    measuredBody.workingMemory.encodedMap.assignRun(
        measuredKey, std::vector<gl::LocalMemoryValue>{ measuredValue });
    measuredBody.localHashMemory.encodedMap.assignRun(
        measuredKey, std::vector<gl::LocalMemoryValue>{ measuredValue });
    measuredBody.localHashMemoryDelta.encodedMap.assignRun(
        measuredKey, std::vector<gl::LocalMemoryValue>{ measuredValue });
    const gl::gpu::Phase2TaskProjectionUsage measured =
        gl::gpu::measurePhase2TaskProjectionUsage(measuredBody, 2, false);
    ASSERT_EQ(measured.tasks, 1u);
    ASSERT_EQ(measured.batches, 4u);
    ASSERT_EQ(measured.terms, 4u);
    ASSERT_EQ(measured.stumps, 2u);
    const gl::Memory emptyBody;
    const gl::gpu::Phase2TaskProjectionUsage minimal =
        gl::gpu::measurePhase2TaskProjectionUsage(emptyBody, 0, false);
    ASSERT_EQ(minimal.tasks, 0u);
    ASSERT_EQ(minimal.batches, 0u);
    ASSERT_EQ(minimal.terms, 0u);
    ASSERT_EQ(minimal.stumps, 0u);
    const gl::gpu::Phase2TaskProjectionUsage counterExampleUsage =
        gl::gpu::measurePhase2TaskProjectionUsage(emptyBody, 0, true);
    ASSERT_EQ(counterExampleUsage.tasks, 1u);
    ASSERT_EQ(counterExampleUsage.batches, 1u);
    ASSERT_EQ(counterExampleUsage.terms, 0u);
    ASSERT_EQ(counterExampleUsage.stumps, 0u);

    gl::gpu::Phase2TaskProjectionCapacity capacity{};
    capacity.tasks = 2;
    capacity.batches = 8;
    capacity.terms = 8;
    capacity.stumps = 4;
    gl::gpu::Phase2TaskProjectionArena arena(capacity);

    gl::gpu::Phase2RequestBatchInput inputs[4]{};
    inputs[0].kind = gl::gpu::DeviceRequestBatchKind::workingRules;
    inputs[0].memory = gl::gpu::DeviceHashMemoryKind::working;
    inputs[0].terms[0].views[0] =
        gl::gpu::DeviceMandatoryViewKind::local;
    inputs[0].terms[0].viewCount = 1;
    inputs[0].termCount = 1;

    inputs[1].kind = gl::gpu::DeviceRequestBatchKind::newThisBurst;
    inputs[1].memory = gl::gpu::DeviceHashMemoryKind::overall;
    inputs[1].terms[0].views[0] =
        gl::gpu::DeviceMandatoryViewKind::localDelta;
    inputs[1].terms[0].viewCount = 1;
    inputs[1].terms[1].views[0] =
        gl::gpu::DeviceMandatoryViewKind::external;
    inputs[1].terms[1].views[1] =
        gl::gpu::DeviceMandatoryViewKind::local;
    inputs[1].terms[1].viewCount = 2;
    inputs[1].termCount = 2;

    inputs[2].kind = gl::gpu::DeviceRequestBatchKind::localRulesWithMail;
    inputs[2].memory = gl::gpu::DeviceHashMemoryKind::local;
    inputs[2].terms[0].views[0] =
        gl::gpu::DeviceMandatoryViewKind::external;
    inputs[2].terms[0].viewCount = 1;
    inputs[2].termCount = 1;

    inputs[3].kind = gl::gpu::DeviceRequestBatchKind::localDeltaRules;
    inputs[3].memory = gl::gpu::DeviceHashMemoryKind::localDelta;
    inputs[3].termCount = 0;

    gl::ExpressionStump sourceStumps[2]{};
    sourceStumps[0].allIdx[0] = 7;
    sourceStumps[0].count = 1;
    sourceStumps[0].terminalOnly = 1;
    sourceStumps[1].allIdx[0] = 2;
    sourceStumps[1].allIdx[1] = 9;
    sourceStumps[1].count = 2;
    sourceStumps[1].terminalOnly = 0;

    const gl::gpu::DevicePhase2Task task = arena.appendTask(
        5, inputs, 4, sourceStumps, 2, 1, 3, 12, 0);
    ASSERT_EQ(task.logicalBlockIndex, 5u);
    ASSERT_EQ(task.batchOffset, 0u);
    ASSERT_EQ(task.batchCount, 4u);
    ASSERT_EQ(task.stumpOffset, 0u);
    ASSERT_EQ(task.stumpCount, 2u);
    ASSERT_EQ(task.stumpOrdinal, 1);
    ASSERT_EQ(task.stumpTotal, 3);
    ASSERT_EQ(task.maximumIterationNumberVariable, 12);
    ASSERT_EQ(task.counterExampleMode, 0u);
    ASSERT_EQ(arena.batches[0].kind,
              gl::gpu::DeviceRequestBatchKind::workingRules);
    ASSERT_EQ(arena.batches[0].termOffset, 0u);
    ASSERT_EQ(arena.batches[0].termCount, 1u);
    ASSERT_EQ(arena.batches[1].termOffset, 1u);
    ASSERT_EQ(arena.batches[1].termCount, 2u);
    ASSERT_EQ(arena.batches[2].termOffset, 3u);
    ASSERT_EQ(arena.batches[2].termCount, 1u);
    ASSERT_EQ(arena.batches[3].termOffset, 4u);
    ASSERT_EQ(arena.batches[3].termCount, 0u);
    ASSERT_EQ(arena.terms.size(), 4u);
    ASSERT_EQ(arena.terms[2].viewCount, 2u);
    ASSERT_EQ(arena.terms[2].views[0],
              gl::gpu::DeviceMandatoryViewKind::external);
    ASSERT_EQ(arena.terms[2].views[1],
              gl::gpu::DeviceMandatoryViewKind::local);
    ASSERT_EQ(arena.stumps[0].statementIndices[0], 7);
    ASSERT_EQ(arena.stumps[0].terminalOnly, 1u);
    ASSERT_EQ(arena.stumps[1].statementIndices[0], 2);
    ASSERT_EQ(arena.stumps[1].statementIndices[1], 9);

    const gl::gpu::DevicePhase2Task* taskAllocation = arena.tasks.data();
    const gl::gpu::DeviceRequestBatch* batchAllocation = arena.batches.data();
    const gl::gpu::DeviceMandatoryTerm* termAllocation = arena.terms.data();
    const gl::gpu::DeviceExpressionStump* stumpAllocation = arena.stumps.data();
    arena.clear();
    ASSERT_EQ(arena.tasks.data(), taskAllocation);
    ASSERT_EQ(arena.batches.data(), batchAllocation);
    ASSERT_EQ(arena.terms.data(), termAllocation);
    ASSERT_EQ(arena.stumps.data(), stumpAllocation);

    gl::gpu::Phase2RequestBatchInput counterExample{};
    counterExample.kind = gl::gpu::DeviceRequestBatchKind::counterExample;
    counterExample.memory = gl::gpu::DeviceHashMemoryKind::overall;
    const gl::gpu::DevicePhase2Task reused = arena.appendTask(
        0, &counterExample, 1, nullptr, 0, 0, 0, 12, 1);
    ASSERT_EQ(reused.batchOffset, 0u);
    ASSERT_EQ(reused.stumpCount, 0u);
    ASSERT_EQ(reused.counterExampleMode, 1u);
    ASSERT_EQ(arena.batches[0].termCount, 0u);

    gl::gpu::Phase2FilterScheduleCapacity filterCapacity{};
    filterCapacity.calls = 4;
    filterCapacity.examinedRows = 64;
    filterCapacity.retainedRows = 32;
    filterCapacity.maximumExaminedRowsPerCall = 32;
    gl::gpu::Phase2FilterScheduleArena filterSchedule(filterCapacity);
    const gl::gpu::DevicePhase2FilterCall filterCall =
        filterSchedule.appendCall(
            3, gl::gpu::DeviceHashMemoryKind::localDelta,
            12, 1, 17);
    ASSERT_EQ(filterCall.logicalBlockIndex, 3u);
    ASSERT_EQ(filterCall.memory,
              gl::gpu::DeviceHashMemoryKind::localDelta);
    ASSERT_EQ(filterCall.maximumIterationNumberVariable, 12);
    ASSERT_EQ(filterCall.alsoAcceptFullKeys, 1u);
    ASSERT_EQ(filterCall.statementCount, 17u);
    ASSERT_EQ(filterSchedule.examinedRows, 17u);
    ASSERT_EQ(filterSchedule.classes.size(), 1u);
    ASSERT_EQ(filterSchedule.callClassIndices.size(), 1u);
    ASSERT_EQ(filterSchedule.callClassIndices[0], 0u);
    ASSERT_EQ(filterSchedule.classMultiplicities[0], 1u);
    ASSERT_EQ(filterSchedule.classExaminedRows, 17u);
    const gl::gpu::DevicePhase2FilterCall* filterAllocation =
        filterSchedule.calls.data();
    const gl::gpu::DevicePhase2FilterCall* classAllocation =
        filterSchedule.classes.data();
    const uint32_t* classIndexAllocation =
        filterSchedule.callClassIndices.data();
    const uint32_t* multiplicityAllocation =
        filterSchedule.classMultiplicities.data();
    const int32_t* classSlotAllocation = filterSchedule.classSlots.data();
    filterSchedule.clear();
    ASSERT_EQ(filterSchedule.calls.data(), filterAllocation);
    ASSERT_EQ(filterSchedule.classes.data(), classAllocation);
    ASSERT_EQ(filterSchedule.callClassIndices.data(), classIndexAllocation);
    ASSERT_EQ(filterSchedule.classMultiplicities.data(), multiplicityAllocation);
    ASSERT_EQ(filterSchedule.classSlots.data(), classSlotAllocation);
    ASSERT_EQ(filterSchedule.calls.size(), 0u);
    ASSERT_EQ(filterSchedule.classes.size(), 0u);
    ASSERT_EQ(filterSchedule.callClassIndices.size(), 0u);
    ASSERT_EQ(filterSchedule.classMultiplicities.size(), 0u);
    ASSERT_EQ(filterSchedule.examinedRows, 0u);
    ASSERT_EQ(filterSchedule.classExaminedRows, 0u);
    for (int32_t slot : filterSchedule.classSlots) ASSERT_EQ(slot, -1);
}

TEST(phase2_projection, filter_class_census_uses_the_complete_exact_tuple) {
    gl::gpu::Phase2FilterScheduleCapacity capacity{};
    capacity.calls = 8;
    capacity.examinedRows = 128;
    capacity.retainedRows = 64;
    capacity.maximumExaminedRowsPerCall = 32;
    gl::gpu::Phase2FilterScheduleArena schedule(capacity);

    schedule.appendCall(
        3, gl::gpu::DeviceHashMemoryKind::overall, 12, 1, 17);
    schedule.appendCall(
        3, gl::gpu::DeviceHashMemoryKind::overall, 12, 1, 17);
    schedule.appendCall(
        4, gl::gpu::DeviceHashMemoryKind::overall, 12, 1, 17);
    schedule.appendCall(
        3, gl::gpu::DeviceHashMemoryKind::local, 12, 1, 17);
    schedule.appendCall(
        3, gl::gpu::DeviceHashMemoryKind::overall, 13, 1, 17);
    schedule.appendCall(
        3, gl::gpu::DeviceHashMemoryKind::overall, 12, 0, 17);
    schedule.appendCall(
        3, gl::gpu::DeviceHashMemoryKind::overall, 12, 1, 16);

    const gl::gpu::Phase2FilterClassCensus census =
        schedule.measureClassReuse();
    ASSERT_EQ(census.callCount, 7u);
    ASSERT_EQ(census.uniqueClassCount, 6u);
    ASSERT_EQ(census.duplicateCallCount, 1u);
    ASSERT_EQ(census.uniqueExaminedRows, 101u);
    ASSERT_EQ(census.duplicateExaminedRows, 17u);
    ASSERT_EQ(census.maximumClassMultiplicity, 2u);
    ASSERT_EQ(schedule.calls.size(), 7u);
    ASSERT_EQ(schedule.examinedRows, 118u);
    ASSERT_EQ(schedule.classes.size(), 6u);
    ASSERT_EQ(schedule.callClassIndices.size(), 7u);
    ASSERT_EQ(schedule.callClassIndices[0], 0u);
    ASSERT_EQ(schedule.callClassIndices[1], 0u);
    ASSERT_EQ(schedule.callClassIndices[2], 1u);
    ASSERT_EQ(schedule.callClassIndices[3], 2u);
    ASSERT_EQ(schedule.callClassIndices[4], 3u);
    ASSERT_EQ(schedule.callClassIndices[5], 4u);
    ASSERT_EQ(schedule.callClassIndices[6], 5u);
    ASSERT_EQ(schedule.classMultiplicities[0], 2u);
    for (uint32_t classIndex = 1; classIndex < 6; ++classIndex) {
        ASSERT_EQ(schedule.classMultiplicities[classIndex], 1u);
    }
    ASSERT_EQ(schedule.classExaminedRows, 101u);

    schedule.clear();
    const gl::gpu::Phase2FilterClassCensus empty =
        schedule.measureClassReuse();
    ASSERT_EQ(empty.callCount, 0u);
    ASSERT_EQ(empty.uniqueClassCount, 0u);
    ASSERT_EQ(empty.duplicateCallCount, 0u);
    ASSERT_EQ(empty.uniqueExaminedRows, 0u);
    ASSERT_EQ(empty.duplicateExaminedRows, 0u);
    ASSERT_EQ(empty.maximumClassMultiplicity, 0u);
}

#ifdef GL_CUDA
TEST(phase2_projection, fixed_cuda_task_buffer_upload_is_exact) {
    gl::gpu::Phase2TaskProjectionCapacity capacity{};
    capacity.tasks = 2;
    capacity.batches = 4;
    capacity.terms = 4;
    capacity.stumps = 4;
    gl::gpu::Phase2TaskProjectionArena host(capacity);

    gl::gpu::Phase2RequestBatchInput batches[2]{};
    batches[0].kind = gl::gpu::DeviceRequestBatchKind::workingRules;
    batches[0].memory = gl::gpu::DeviceHashMemoryKind::working;
    batches[0].terms[0].views[0] =
        gl::gpu::DeviceMandatoryViewKind::local;
    batches[0].terms[0].viewCount = 1;
    batches[0].termCount = 1;
    batches[1].kind = gl::gpu::DeviceRequestBatchKind::newThisBurst;
    batches[1].memory = gl::gpu::DeviceHashMemoryKind::overall;
    batches[1].terms[0].views[0] =
        gl::gpu::DeviceMandatoryViewKind::localDelta;
    batches[1].terms[0].views[1] =
        gl::gpu::DeviceMandatoryViewKind::external;
    batches[1].terms[0].viewCount = 2;
    batches[1].termCount = 1;

    gl::ExpressionStump stumps[2]{};
    stumps[0].allIdx[0] = 3;
    stumps[0].count = 1;
    stumps[0].terminalOnly = 1;
    stumps[1].allIdx[0] = 5;
    stumps[1].allIdx[1] = 8;
    stumps[1].count = 2;
    host.appendTask(7, batches, 2, stumps, 2, 0, 2, 14, 0);

    gl::gpu::CudaPhase2TaskBuffer device(capacity);
    device.upload(host);
    ASSERT_EQ(device.launchChecksum(), taskProjectionChecksum(host));
}

TEST(phase2_projection, bulk_cuda_filter_and_sort_matches_processor_streams) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.maxIterationNumberVariable = 3;
    gl::Memory body;
    gl::NameMap& names = body.nameMap;
    const gl::NameId mainId = names.encode("main");
    const gl::NameId frozenBranch = names.encodePush(
        mainId, gl::StrSpan("ordis_(filter-test)", 19));
    const int32_t fixtureRuleId =
        body.ruleInterner.encode("(fixture-rule[1])");
    const gl::RuleOwner fixtureOwner =
        gl::packRuleOwner(fixtureRuleId, mainId);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    const std::string expressions[6] = {
        "(z[a])", "(a[b])", "(a[c])", "(m[d])", "(q[e])", "(b[f])"
    };
    for (uint32_t index = 0; index < 6; ++index) {
        const gl::NameId validity = index == 4 ? frozenBranch : mainId;
        gl::IntEncodedExpr statement = gl::encodeExpression(
            gl::EncodedExpression(expressions[index], names.decode(validity)),
            names);
        if (index == 5) statement.maxIteration = 4;
        body.intEncodedStatements.push_back(statement);
    }
    body.frozenOrBranches.mint(frozenBranch);

    gl::HashMemory* memories[4] = {
        &body.overallHashMemory, &body.localHashMemory,
        &body.localHashMemoryDelta, &body.workingMemory
    };
    for (uint32_t statementIndex = 0; statementIndex < 6; ++statementIndex) {
        const gl::IntEncodedExpr& statement =
            body.intEncodedStatements[statementIndex];
        gl::NameId keyData[4] = {
            statement.nameId, statement.negation, 1, 0
        };
        const gl::NormKey key{
            1, std::vector<gl::NameId>(keyData, keyData + 4)
        };
        for (gl::HashMemory* memory : memories) {
            if (statementIndex == 3) {
                gl::ExpressionAnalyzer::addWholeKeyOwner(
                    memory->normalizedEncodedKeys, 1, keyData, 4,
                    fixtureOwner, ownerArena);
            }
            else {
                gl::ExpressionAnalyzer::addShortSubkeyOwner(
                    memory->normalizedEncodedSubkeys, 1, keyData, 4,
                    fixtureOwner);
            }
        }
    }
    std::clog << "[phase2-filter-twin] fixture ready\n" << std::flush;

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= names.nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(names.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity projectionLimits =
        projectionCapacity(
            1, static_cast<uint32_t>(body.intEncodedStatements.size()),
            static_cast<uint32_t>(names.nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena hostProjection(projectionLimits);
    const gl::gpu::DeviceLogicalBlockProjection block =
        hostProjection.appendLogicalBlock(body, analyzer);
    std::clog << "[phase2-filter-twin] projection packed\n" << std::flush;
    gl::gpu::CudaPhase2ProjectionBuffer deviceProjection(projectionLimits);
    deviceProjection.upload(hostProjection);
    std::clog << "[phase2-filter-twin] projection uploaded\n" << std::flush;

    const auto processorOrder = [&](const char* label,
                                    gl::HashMemory& memory,
                                    bool alsoAcceptFullKeys) {
        std::clog << "[phase2-filter-twin] processor " << label
                  << " start\n" << std::flush;
        gl::NameId output[8192]{};
        const gl::NameId count = analyzer.filterIntEncodedStatements(
            gl::IntStmtView(body.intEncodedStatements), memory, body,
            alsoAcceptFullKeys, output, 8192);
        std::vector<gl::NameId> result(output, output + count);
        std::stable_sort(result.begin(), result.end(),
            [&](gl::NameId left, gl::NameId right) {
                return gl::compareSpans(
                    names.decodeView(body.intEncodedStatements[left].nameId),
                    names.decodeView(body.intEncodedStatements[right].nameId)) < 0;
            });
        std::clog << "[phase2-filter-twin] processor " << label
                  << " complete\n" << std::flush;
        return result;
    };
    const std::vector<gl::NameId> expected[5] = {
        processorOrder("overall-subkey", body.overallHashMemory, false),
        processorOrder("overall-union", body.overallHashMemory, true),
        processorOrder("local", body.localHashMemory, true),
        processorOrder("local-delta", body.localHashMemoryDelta, true),
        processorOrder("working", body.workingMemory, true)
    };
    std::clog << "[phase2-filter-twin] processor oracle complete\n"
              << std::flush;

    gl::gpu::Phase2FilterScheduleCapacity filterLimits{};
    filterLimits.calls = 2048;
    filterLimits.examinedRows = 16777216;
    filterLimits.retainedRows = 1048576;
    filterLimits.maximumExaminedRowsPerCall = 262144;
    gl::gpu::Phase2FilterScheduleArena schedule(filterLimits);
    const gl::gpu::DeviceHashMemoryKind kinds[5] = {
        gl::gpu::DeviceHashMemoryKind::overall,
        gl::gpu::DeviceHashMemoryKind::overall,
        gl::gpu::DeviceHashMemoryKind::local,
        gl::gpu::DeviceHashMemoryKind::localDelta,
        gl::gpu::DeviceHashMemoryKind::working
    };
    for (uint32_t call = 0; call < 5; ++call) {
        schedule.appendCall(
            0, kinds[call], analyzer.parameters.maxIterationNumberVariable,
            call == 0 ? 0u : 1u, block.statementCount);
    }

    gl::gpu::CudaPhase2FilterSortBuffer deviceFilter(filterLimits);
    const uint32_t retained = deviceFilter.filterAndSort(
        deviceProjection, schedule);
    std::clog << "[phase2-filter-twin] CUDA filter complete\n" << std::flush;
    std::vector<uint32_t> counts(schedule.calls.size());
    ASSERT_EQ(deviceFilter.downloadCallCounts(
        counts.data(), static_cast<uint32_t>(counts.size())),
        static_cast<uint32_t>(counts.size()));
    std::vector<uint64_t> sortedKeys(retained);
    ASSERT_EQ(deviceFilter.downloadSortedKeys(
        sortedKeys.data(), static_cast<uint32_t>(sortedKeys.size())), retained);

    uint32_t keyOffset = 0;
    for (uint32_t call = 0; call < 5; ++call) {
        ASSERT_EQ(counts[call], static_cast<uint32_t>(expected[call].size()));
        for (uint32_t row = 0; row < counts[call]; ++row) {
            const uint64_t key = sortedKeys[keyOffset++];
            ASSERT_EQ(static_cast<uint32_t>(
                key >> gl::gpu::kDeviceFilterCallShift), call);
            ASSERT_EQ(static_cast<gl::NameId>(
                key & gl::gpu::kDeviceFilterStatementIndexMask),
                expected[call][row]);
        }
    }
    ASSERT_EQ(keyOffset, retained);
}

TEST(phase2_projection, fixed_cuda_buffer_upload_and_lookups_are_exact) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.compiledExpressions["delta"] = gl::LogicalEntity(
        "atomic", {}, "delta", 1);
    gl::Memory body;
    const gl::NameId mainId = body.nameMap.encode("main");
    const gl::NameId nameId = body.nameMap.encode("delta");
    const gl::NameId originalId = body.nameMap.encode("(delta[delta])");
    const gl::IntEncodedExpr statement =
        projectionStatement(nameId, originalId, mainId, nameId);
    body.intEncodedStatements.push_back(statement);
    body.intLocalEncodedStatements.push_back(statement);
    body.intLocalEncodedStatementsDelta.push_back(statement);
    body.intExternalStatements.push_back(statement);
    body.setExprKey("cuda_projection");

    const int32_t ruleId = body.ruleInterner.encode("(delta[1])");
    const gl::NormKey normKey{ 1, { nameId, mainId } };
    gl::OwnerSet owner;
    owner.hasLooseOwner = true;
    const gl::RuleOwner packedOwner = gl::packRuleOwner(ruleId, mainId);
    owner.owners.emplace_back(packedOwner, -1);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    gl::LocalMemoryValue value;
    value.valueId = ruleId;
    value.originalImplicationId = ruleId;
    value.validityId = mainId;
    const gl::Int16SetKey remainingKey{ { nameId } };
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.overallHashMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    body.overallHashMemory.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    body.overallHashMemory.encodedMap.assignRun(
        normKey, std::vector<gl::LocalMemoryValue>{ value });
    body.overallHashMemory.remainingArgsNormalizedEncodedMap.assignRun(
        remainingKey, std::vector<gl::NormKey>{ normKey });
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.localHashMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    body.localHashMemory.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.localHashMemoryDelta.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    body.localHashMemoryDelta.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.workingMemory.normalizedEncodedKeys,
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    body.workingMemory.normalizedEncodedSubkeys.assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    body.overallHashMemory.productsOfRecursionIds.mint(nameId);
    const gl::StatementKey statementKey{ originalId, mainId };
    const int64_t packedStatementKey =
        gl::packStatementKey(originalId, mainId);
    body.intKnownStatements.upsert(
        statementKey, gl::StatementFlags{ true, true });
    body.intLocalEncodedStatementsSet.mint(packedStatementKey);
    const int levels[1] = { 4 };
    body.intStatementLevelsMap.assignSet(statementKey, levels, 1);
    body.intValidityNamesToFilter.mint(mainId);
    body.frozenOrBranches.mint(nameId);
    body.intToBeProved.assignSet(statementKey, levels, 1);
    body.canBeSentIds.mint(nameId);
    body.canBeSentMarkerIds.mint(nameId);

    uint32_t byteCount = 0;
    for (gl::NameId id = 1; id <= body.nameMap.nameCount(); ++id)
        byteCount += static_cast<uint32_t>(body.nameMap.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity capacity = projectionCapacity(
        1,
        1,
        static_cast<uint32_t>(body.nameMap.nameCount() + 1),
        byteCount);
    gl::gpu::Phase2ProjectionArena host(capacity);
    const gl::gpu::DeviceLogicalBlockProjection projected =
        host.appendLogicalBlock(body, analyzer);
    ASSERT_TRUE(!host.byteMapEntries.empty());
    ASSERT_TRUE(!host.blobRecords.empty());
    ASSERT_TRUE(!host.reverseMapEntries.empty());
    ASSERT_TRUE(!host.podMapEntries.empty());
    ASSERT_TRUE(!host.podRunValues.empty());
    ASSERT_TRUE(!host.mandatoryStatementKeys.empty());
    ASSERT_TRUE(!host.metadataBytes.empty());

    gl::gpu::CudaPhase2ProjectionBuffer device(capacity);
    device.upload(host);
    ASSERT_EQ(device.launchChecksum(), projectionChecksum(host));
    device.beginPhase2Upload(host);
    device.finishPhase2Upload();
    ASSERT_EQ(device.launchChecksum(), projectionChecksum(host));

    const auto byteProbe = [](
        gl::gpu::DeviceLookupProbeKind probeKind,
        uint32_t mapKind,
        const std::string& key) {
        gl::gpu::DeviceLookupProbe probe{};
        probe.kind = probeKind;
        probe.mapKind = mapKind;
        ASSERT_TRUE(key.size()
            <= gl::gpu::DeviceLookupProbe::kMaximumKeyBytes);
        probe.keyLength = static_cast<uint32_t>(key.size());
        std::memcpy(probe.key, key.data(), key.size());
        return probe;
    };

    gl::gpu::DeviceLookupProbe nameProbe = byteProbe(
        gl::gpu::DeviceLookupProbeKind::name, 0, "delta");
    const gl::gpu::DeviceLookupProbeResult nameHit =
        device.launchLookupProbe(nameProbe);
    ASSERT_EQ(nameHit.recordIndex, nameId);
    ASSERT_EQ(nameHit.payloadCount, 5u);
    ASSERT_EQ(nameHit.firstIntValue, body.nameMap.parentOf(nameId));
    ASSERT_EQ(nameHit.scalar, host.nameRecords[
        projected.nameRecordOffset + static_cast<uint32_t>(nameId)].decodedLexRank);
    nameProbe = byteProbe(
        gl::gpu::DeviceLookupProbeKind::name, 0, "not-a-projected-name");
    ASSERT_EQ(device.launchLookupProbe(nameProbe).recordIndex, -1);

    const std::string normBytes = gl::Codec<gl::NormKey>::encode(normKey);
    const std::string remainingBytes =
        gl::Codec<gl::Int16SetKey>::encode(remainingKey);
    for (uint32_t ordinal = 0;
         ordinal <= static_cast<uint32_t>(
             gl::gpu::DeviceByteMapKind::overallRemainingArgs);
         ++ordinal) {
        const auto kind = static_cast<gl::gpu::DeviceByteMapKind>(ordinal);
        const std::string& key = kind
            == gl::gpu::DeviceByteMapKind::overallRemainingArgs
            ? remainingBytes : normBytes;
        const gl::gpu::DeviceLookupProbeResult hit = device.launchLookupProbe(
            byteProbe(gl::gpu::DeviceLookupProbeKind::byteMap, ordinal, key));
        const gl::gpu::DeviceByteMapView& view =
            host.byteMapViews[projected.byteMapViewOffset + ordinal];
        ASSERT_EQ(hit.recordIndex, static_cast<int32_t>(view.entryOffset));
        const bool hasPayload = kind == gl::gpu::DeviceByteMapKind::overallSubkeys
            || kind == gl::gpu::DeviceByteMapKind::localSubkeys
            || kind == gl::gpu::DeviceByteMapKind::deltaSubkeys
            || kind == gl::gpu::DeviceByteMapKind::workingSubkeys
            || kind == gl::gpu::DeviceByteMapKind::overallEncoded
            || kind == gl::gpu::DeviceByteMapKind::overallRemainingArgs;
        ASSERT_EQ(hit.payloadCount, hasPayload ? 1u : 0u);
        if (hasPayload) {
            const gl::gpu::DeviceByteMapEntry& entry =
                host.byteMapEntries[view.entryOffset];
            ASSERT_EQ(hit.payloadOffset, entry.blobRecordOffset);
            ASSERT_EQ(hit.firstIntValue, static_cast<int32_t>(
                host.blobRecords[entry.blobRecordOffset].byteLength));
        }
    }
    ASSERT_EQ(device.launchLookupProbe(byteProbe(
        gl::gpu::DeviceLookupProbeKind::byteMap,
        static_cast<uint32_t>(gl::gpu::DeviceByteMapKind::overallWholeKeys),
        "missing-byte-key")).recordIndex, -1);

    const gl::gpu::DeviceLookupProbeResult reverseHit = device.launchLookupProbe(
        byteProbe(gl::gpu::DeviceLookupProbeKind::reverseMap, 0, normBytes));
    const gl::gpu::DeviceReverseMapView& reverseView =
        host.reverseMapViews[projected.reverseMapViewOffset];
    ASSERT_EQ(reverseHit.recordIndex,
              static_cast<int32_t>(reverseView.entryOffset));
    ASSERT_EQ(reverseHit.payloadCount, 1u);
    ASSERT_EQ(reverseHit.firstIntValue, 1);
    ASSERT_EQ(device.launchLookupProbe(byteProbe(
        gl::gpu::DeviceLookupProbeKind::reverseMap, 0,
        "missing-reverse-key")).recordIndex, -1);

    const int64_t podKeys[11] = {
        static_cast<int64_t>(nameId),
        packedStatementKey,
        packedStatementKey,
        packedStatementKey,
        packedStatementKey,
        packedStatementKey,
        static_cast<int64_t>(mainId),
        static_cast<int64_t>(nameId),
        packedStatementKey,
        static_cast<int64_t>(nameId),
        static_cast<int64_t>(nameId)
    };
    for (uint32_t ordinal = 0;
         ordinal <= static_cast<uint32_t>(
             gl::gpu::DevicePodMapKind::mailEligibleMarkers);
         ++ordinal) {
        gl::gpu::DeviceLookupProbe probe{};
        probe.kind = gl::gpu::DeviceLookupProbeKind::podMap;
        probe.mapKind = ordinal;
        probe.podKey = podKeys[ordinal];
        const gl::gpu::DeviceLookupProbeResult hit =
            device.launchLookupProbe(probe);
        const gl::gpu::DevicePodMapView& view =
            host.podMapViews[projected.podMapViewOffset + ordinal];
        ASSERT_EQ(hit.recordIndex, static_cast<int32_t>(view.entryOffset));
        const gl::gpu::DevicePodMapEntry& entry =
            host.podMapEntries[view.entryOffset];
        ASSERT_EQ(hit.payloadOffset, entry.runOffset);
        ASSERT_EQ(hit.payloadCount, entry.runCount);
        ASSERT_EQ(hit.scalar, entry.scalar);
        if (entry.runCount > 0)
            ASSERT_EQ(hit.firstIntValue, host.podRunValues[entry.runOffset]);
    }
    gl::gpu::DeviceLookupProbe podMiss{};
    podMiss.kind = gl::gpu::DeviceLookupProbeKind::podMap;
    podMiss.mapKind = static_cast<uint32_t>(
        gl::gpu::DevicePodMapKind::knownStatements);
    podMiss.podKey = -9223372036854775807ll;
    ASSERT_EQ(device.launchLookupProbe(podMiss).recordIndex, -1);
}

TEST(phase2_projection, cuda_normalized_key_and_owner_gate_match_processor) {
    gl::Memory body;
    body.setExprKey("cuda_growth_candidate");
    gl::ExpressionAnalyzer analyzer("Peano");
    const gl::NameId mainId = body.nameMap.encode("main");
    const gl::NameId alphaId = body.nameMap.encode("alpha");
    const gl::NameId betaId = body.nameMap.encode("beta");
    const gl::NameId gammaId = body.nameMap.encode("gamma");
    const gl::NameId patternX = body.nameMap.encode("x");
    const gl::NameId patternY = body.nameMap.encode("y");
    const gl::NameId literalA = body.nameMap.encode("u_literal_a");
    const gl::NameId literalB = body.nameMap.encode("u_literal_b");
    const gl::NameId wrongLiteral = body.nameMap.encode("u_wrong");

    gl::IntEncodedExpr first = projectionStatement(
        alphaId, alphaId, mainId, patternX);
    first.argFullId[0] = literalA;
    gl::IntEncodedExpr second = projectionStatement(
        betaId, betaId, mainId, patternX);
    second.negation = 1;
    second.argFullId[0] = literalA;
    gl::IntEncodedExpr third = projectionStatement(
        gammaId, gammaId, mainId, patternY);
    third.argFullId[0] = literalB;
    body.intEncodedStatements.push_back(first);
    body.intEncodedStatements.push_back(second);
    body.intEncodedStatements.push_back(third);

    const gl::IntEncodedExpr* expressions[3] = {
        &body.intEncodedStatements[0],
        &body.intEncodedStatements[1],
        &body.intEncodedStatements[2]
    };
    gl::NameId expectedData[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
    const gl::NameId expectedLength = analyzer.makeIntNormalizedKeyFromEncoded(
        expressions, 3, expectedData,
        gl::ExecutionParameters::MAX_KEY_SLOTS);
    const gl::NormKey key{
        3, std::vector<gl::NameId>(
            expectedData, expectedData + expectedLength) };

    gl::OwnerSet acceptingOwner;
    acceptingOwner.uSignatures.insert({
        { 0, literalA }, { 1, literalA }, { 2, literalB } });
    gl::OwnerSet rejectingOwner;
    rejectingOwner.uSignatures.insert({ { 0, wrongLiteral } });
    const int32_t acceptingRuleId = body.ruleInterner.encode(
        "(cuda-accepting-rule[1])");
    const int32_t rejectingRuleId = body.ruleInterner.encode(
        "(cuda-rejecting-rule[1])");
    const gl::RuleOwner acceptingRuleOwner = gl::packRuleOwner(
        acceptingRuleId, mainId);
    const gl::RuleOwner rejectingRuleOwner = gl::packRuleOwner(
        rejectingRuleId, mainId);
    acceptingOwner.owners.emplace_back(acceptingRuleOwner, 0);
    rejectingOwner.owners.emplace_back(rejectingRuleOwner, 0);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    ASSERT_TRUE(gl::ownerSetUSatisfied(acceptingOwner, expressions, 3));
    ASSERT_TRUE(!gl::ownerSetUSatisfied(rejectingOwner, expressions, 3));

    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.overallHashMemory.normalizedEncodedKeys,
        key.numberExpressions, key.data.data(),
        static_cast<int32_t>(key.data.size()), acceptingRuleOwner, ownerArena);
    body.overallHashMemory.normalizedEncodedSubkeys.assignRun(
        key, std::vector<gl::OwnerSet>{ acceptingOwner });
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.localHashMemory.normalizedEncodedKeys,
        key.numberExpressions, key.data.data(),
        static_cast<int32_t>(key.data.size()), rejectingRuleOwner, ownerArena);
    body.localHashMemory.normalizedEncodedSubkeys.assignRun(
        key, std::vector<gl::OwnerSet>{ rejectingOwner });

    gl::NameId shortData[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
    const gl::NameId shortLength = analyzer.makeIntNormalizedKeyFromEncoded(
        expressions, 2, shortData,
        gl::ExecutionParameters::MAX_KEY_SLOTS);
    const gl::NormKey shortKey{
        2, std::vector<gl::NameId>(shortData, shortData + shortLength) };
    body.localHashMemoryDelta.normalizedEncodedSubkeys.assignRun(
        shortKey, std::vector<gl::OwnerSet>{ rejectingOwner });

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= body.nameMap.nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(body.nameMap.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity capacity = projectionCapacity(
        1, static_cast<uint32_t>(body.intEncodedStatements.size()),
        static_cast<uint32_t>(body.nameMap.nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena host(capacity);
    host.appendLogicalBlock(body, analyzer);
    gl::gpu::CudaPhase2ProjectionBuffer device(capacity);
    device.upload(host);

    gl::gpu::DeviceGrowthCandidateProbe probe{};
    probe.count = 3;
    probe.statementIndices[0] = 0;
    probe.statementIndices[1] = 1;
    probe.statementIndices[2] = 2;
    const auto compareKey = [&](const auto& result,
                                const gl::NameId* expected,
                                gl::NameId length) {
        ASSERT_EQ(result.normalizedKeyLength, length);
        for (gl::NameId index = 0; index < length; ++index)
            ASSERT_EQ(result.normalizedKey[index], expected[index]);
    };

    probe.memory = gl::gpu::DeviceHashMemoryKind::overall;
    const gl::gpu::DeviceGrowthCandidateProbeResult accepted =
        device.launchGrowthCandidateProbe(probe);
    compareKey(accepted, expectedData, expectedLength);
    ASSERT_EQ(accepted.subkeyPresent, 1u);
    ASSERT_EQ(accepted.subkeySatisfied, 1u);
    ASSERT_EQ(accepted.wholeKeyPresent, 1u);

    probe.memory = gl::gpu::DeviceHashMemoryKind::local;
    const gl::gpu::DeviceGrowthCandidateProbeResult rejected =
        device.launchGrowthCandidateProbe(probe);
    compareKey(rejected, expectedData, expectedLength);
    ASSERT_EQ(rejected.subkeyPresent, 1u);
    ASSERT_EQ(rejected.subkeySatisfied, 0u);
    ASSERT_EQ(rejected.wholeKeyPresent, 1u);

    probe.memory = gl::gpu::DeviceHashMemoryKind::working;
    const gl::gpu::DeviceGrowthCandidateProbeResult absent =
        device.launchGrowthCandidateProbe(probe);
    ASSERT_EQ(absent.subkeyPresent, 0u);
    ASSERT_EQ(absent.subkeySatisfied, 0u);
    ASSERT_EQ(absent.wholeKeyPresent, 0u);

    probe.memory = gl::gpu::DeviceHashMemoryKind::localDelta;
    probe.count = 2;
    const gl::gpu::DeviceGrowthCandidateProbeResult shortAccepted =
        device.launchGrowthCandidateProbe(probe);
    compareKey(shortAccepted, shortData, shortLength);
    ASSERT_EQ(shortAccepted.subkeyPresent, 1u);
    ASSERT_EQ(shortAccepted.subkeySatisfied, 1u);
    ASSERT_EQ(shortAccepted.wholeKeyPresent, 0u);
}

TEST(phase2_projection, cuda_bulk_growth_matches_request_gate_content) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.maxIterationNumberVariable = 3;
    analyzer.parameters.standardMaxSecondaryNumber = 9;
    analyzer.parameters.maxLenHypoKey = 3;
    analyzer.parameters.maxNumberSecondaryVariables = 1;
    analyzer.parameters.maxNumberSecondaryVariablesOrint = 1;
    analyzer.parameters.incubator_mode = true;
    analyzer.parameters.ban_disintegration = false;
    analyzer.parameters.compressor_mode = false;
    analyzer.compiledExpressions["gpuHead"] = gl::LogicalEntity(
        "atomic", {}, "gpuHead", 2);
    analyzer.compiledExpressions["gpuMarker"] = gl::LogicalEntity(
        "expression", {}, "gpuMarker", 2);
    analyzer.compiledExpressions["c"] = gl::LogicalEntity(
        "expression", {}, "c", 1);

    gl::Memory body;
    body.level = 1;
    body.setExprKey("cuda_bulk_growth");
    gl::NameMap& names = body.nameMap;
    const gl::NameId mainId = names.encode("main");
    const gl::NameId childId = names.encodePush(
        mainId, gl::StrSpan("scope_child", 11));
    const gl::NameId markerOrBranchId = names.encodePush(
        mainId, gl::StrSpan("orint_gpu", 9));
    const gl::NameId siblingId = names.encodePush(
        mainId, gl::StrSpan("scope_sibling", 13));
    const gl::NameId variableProduct = names.encode("product_variable");
    const gl::NameId variableV = names.encode("secondary_v");
    const gl::NameId variableW = names.encode("secondary_w");
    const gl::NameId variableVSecondary = names.encode(
        "secondary_v_shadow");
    const gl::NameId variableWSecondary = names.encode(
        "secondary_w_shadow");
    const char* statementNames[5] = { "a", "b", "c", "d", "e" };
    const gl::NameId validities[5] = {
        mainId, childId, mainId, childId, siblingId
    };
    const gl::NameId arguments[5] = {
        variableProduct, variableV, variableV, variableW, variableV
    };
    for (uint32_t index = 0; index < 5; ++index) {
        const gl::NameId nameId = names.encode(statementNames[index]);
        const std::string original = index == 2
            ? "(c[int_lev_1_2])"
            : std::string("(") + statementNames[index] + "[x])";
        const gl::NameId originalId = names.encode(original);
        gl::IntEncodedExpr statement = projectionStatement(
            nameId, originalId, validities[index], arguments[index]);
        statement.argIteration[0] = 0;
        statement.argFullId[0] = arguments[index];
        if (arguments[index] == variableV) {
            statement.arity = 2;
            statement.argId[1] = index == 4
                ? variableWSecondary : variableVSecondary;
            statement.argIteration[1] = 0;
            statement.argFullId[1] = index == 4
                ? variableWSecondary : variableVSecondary;
        }
        if (index == 0) statement.isAnchor = 1;
        if (index == 2) statement.isAnchor = 1;
        if (index == 1) statement.isHypo = 1;
        body.intEncodedStatements.push_back(statement);
    }
    body.overallHashMemory.productsOfRecursionIds.mint(variableProduct);
    body.overallHashMemory.productsOfRecursionIds.mint(variableV);
    const gl::IntEncodedExpr& mandatory = body.intEncodedStatements[2];
    body.intLocalEncodedStatementsSet.mint(gl::packStatementKey(
        mandatory.originalId, mandatory.validityId));

    const auto normalizedKey = [&](const std::vector<gl::NameId>& indices) {
        const gl::IntEncodedExpr* expressions[
            gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        for (std::size_t index = 0; index < indices.size(); ++index)
            expressions[index] = &body.intEncodedStatements[indices[index]];
        gl::NameId data[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        const gl::NameId length = analyzer.makeIntNormalizedKeyFromEncoded(
            expressions, static_cast<gl::NameId>(indices.size()), data,
            gl::ExecutionParameters::MAX_KEY_SLOTS);
        return gl::NormKey{
            static_cast<gl::NameId>(indices.size()),
            std::vector<gl::NameId>(data, data + length) };
    };
    const int32_t fixtureRuleId = body.ruleInterner.encode(
        "(cuda-growth-rule[1])");
    const gl::RuleOwner fixtureOwner = gl::packRuleOwner(
        fixtureRuleId, mainId);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    const auto addLooseSubkey = [&](const std::vector<gl::NameId>& indices) {
        const gl::NormKey key = normalizedKey(indices);
        gl::OwnerSet owner;
        owner.hasLooseOwner = true;
        owner.owners.emplace_back(fixtureOwner, -1);
        body.overallHashMemory.normalizedEncodedSubkeys.assignRun(
            key, std::vector<gl::OwnerSet>{ owner });
    };
    const auto addWhole = [&](const std::vector<gl::NameId>& indices) {
        const gl::NormKey key = normalizedKey(indices);
        gl::ExpressionAnalyzer::addWholeKeyOwner(
            body.overallHashMemory.normalizedEncodedKeys,
            key.numberExpressions, key.data.data(),
            static_cast<int32_t>(key.data.size()), fixtureOwner, ownerArena);
    };
    for (gl::NameId index = 0; index < 5; ++index)
        addLooseSubkey({ index });
    addLooseSubkey({ 0, 1 });
    addLooseSubkey({ 1, 2 });
    addLooseSubkey({ 2, 3 });
    addLooseSubkey({ 2, 4 });
    gl::OwnerSet rejectingOwner;
    rejectingOwner.uSignatures.insert({ { 0, variableW } });
    rejectingOwner.owners.emplace_back(fixtureOwner, 0);
    body.overallHashMemory.normalizedEncodedSubkeys.assignRun(
        normalizedKey({ 0, 1, 2 }),
        std::vector<gl::OwnerSet>{ rejectingOwner });
    addWhole({ 2 });
    addWhole({ 0, 2 });
    addWhole({ 1, 2 });
    addWhole({ 0, 1, 2 });
    addWhole({ 2, 3 });
    addWhole({ 2, 4 });
    body.overallHashMemory.maxKeyLength = 3;

    const std::vector<std::vector<gl::NameId>> evaluationRequests = {
        { 2 }, { 1, 2 }, { 0, 2 }, { 0, 1, 2 }
    };
    const gl::Int16SetKey remainingV{ { variableV } };
    const gl::Int16SetKey remainingW{ { variableW } };
    std::vector<gl::NormKey> rawEvaluationKeys;
    for (const auto& indices : evaluationRequests)
        rawEvaluationKeys.push_back(normalizedKey(indices));
    body.overallHashMemory.remainingArgsNormalizedEncodedMap.assignRun(
        remainingV, rawEvaluationKeys);
    body.overallHashMemory.remainingArgsNormalizedEncodedMap.assignRun(
        remainingW, std::vector<gl::NormKey>{ rawEvaluationKeys[0] });
    const auto mappedEvaluationKey = [&](const std::vector<gl::NameId>& indices) {
        const gl::IntEncodedExpr* expressions[
            gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        for (std::size_t index = 0; index < indices.size(); ++index)
            expressions[index] = &body.intEncodedStatements[indices[index]];
        gl::NameId data[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        gl::NameId reverse[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        gl::NameId normalizedVariables = 0;
        const gl::NameId length = analyzer.makeIntNormalizedKeyFromEncodedWithMap(
            expressions, static_cast<gl::NameId>(indices.size()),
            std::set<gl::NameId>{ variableV }, data,
            gl::ExecutionParameters::MAX_KEY_SLOTS,
            reverse, normalizedVariables);
        return gl::NormKey{
            static_cast<gl::NameId>(indices.size()),
            std::vector<gl::NameId>(data, data + length) };
    };
    const int32_t gpuHeadId = body.ruleInterner.encode(
        "(gpuHead[1,u_u_static])");
    const int32_t gpuMarkerId = body.ruleInterner.encode(
        "(gpuMarker[marker,1])");
    const int32_t gpuMarkerKeyAId = body.ruleInterner.encode(
        "(gpuKeyA[1,u_u_tail])");
    const int32_t gpuMarkerKeyBId = body.ruleInterner.encode(
        "(gpuKeyB[u_u_static,1])");
    const int32_t gpuRemainingZetaId = body.ruleInterner.encode("zeta");
    const int32_t gpuRemainingAlphaId = body.ruleInterner.encode("alpha");
    const int32_t gpuSourceId = body.ruleInterner.encode(
        "(>[gpuPremise,(gpuHead[1,u_u_static])])");
    for (std::size_t index = 0; index < evaluationRequests.size(); ++index) {
        const uint32_t valueCount = index == 2 ? 2u : 1u;
        std::vector<gl::LocalMemoryValue> values(valueCount);
        for (gl::LocalMemoryValue& value : values) {
            value.valueId = gpuHeadId;
            value.originalImplicationId = gpuSourceId;
            value.justification = gl::RuleJustification::implication;
            value.validityId = mainId;
            value.levels.insert(4);
            value.levels.insert(5);
        }
        if (index == 2) {
            values[1].valueId = gpuMarkerId;
            values[1].validityId = markerOrBranchId;
            values[1].isMarker = true;
            values[1].ordisOnly = true;
            values[1].keyIds = { gpuMarkerKeyAId, gpuMarkerKeyBId };
            values[1].remainingArgIds = {
                gpuRemainingZetaId, gpuRemainingAlphaId,
                gpuRemainingAlphaId };
        }
        body.overallHashMemory.encodedMap.assignRun(
            mappedEvaluationKey(evaluationRequests[index]),
            values);
    }
    const gl::IntEncodedExpr& initiallyKnown = body.intEncodedStatements[2];
    body.intKnownStatements.upsert(
        gl::StatementKey{ initiallyKnown.originalId, initiallyKnown.validityId },
        gl::StatementFlags{ true, false });
    const int initiallyKnownLevels[2] = { -1, 5 };
    body.intStatementLevelsMap.assignSet(
        gl::StatementKey{ initiallyKnown.originalId, initiallyKnown.validityId },
        initiallyKnownLevels, 2);
    const gl::NameId markerMailId = names.encode("(c[marker])");
    body.canBeSentMarkerIds.mint(markerMailId);
    const gl::NameId knownNumericHead = names.encode("(gpuHead[1,static])");
    const gl::NameId knownProductHead = names.encode(
        "(gpuHead[product_variable,static])");
    const gl::NameId knownSecondaryHead = names.encode(
        "(gpuHead[secondary_v_shadow,static])");
    const gl::NameId negatedSecondaryHead = names.encode(
        "!(gpuHead[secondary_v_shadow,static])");
    body.intKnownStatements.upsert(
        gl::StatementKey{ knownNumericHead, mainId },
        gl::StatementFlags{ true, false });
    body.intKnownStatements.upsert(
        gl::StatementKey{ knownProductHead, mainId },
        gl::StatementFlags{ true, false });
    body.intKnownStatements.upsert(
        gl::StatementKey{ knownSecondaryHead, mainId },
        gl::StatementFlags{ true, false });

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= names.nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(names.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity projectionLimits =
        projectionCapacity(
            1, static_cast<uint32_t>(body.intEncodedStatements.size()),
            static_cast<uint32_t>(names.nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena hostProjection(projectionLimits);
    const gl::gpu::DeviceLogicalBlockProjection block =
        hostProjection.appendLogicalBlock(body, analyzer);
    gl::gpu::CudaPhase2ProjectionBuffer deviceProjection(projectionLimits);
    deviceProjection.upload(hostProjection);

    gl::gpu::Phase2RequestBatchInput batch{};
    batch.kind = gl::gpu::DeviceRequestBatchKind::newThisBurst;
    batch.memory = gl::gpu::DeviceHashMemoryKind::overall;
    batch.terms[0].views[0] =
        gl::gpu::DeviceMandatoryViewKind::local;
    batch.terms[0].viewCount = 1;
    batch.termCount = 1;
    gl::gpu::Phase2TaskProjectionCapacity taskLimits{};
    taskLimits.tasks = 4;
    taskLimits.batches = 8;
    taskLimits.terms = 8;
    taskLimits.stumps = 8;
    gl::gpu::Phase2TaskProjectionArena hostTasks(taskLimits);
    hostTasks.appendTask(
        0, &batch, 1, nullptr, 0, 0, 0,
        analyzer.parameters.maxIterationNumberVariable, 0);
    gl::gpu::CudaPhase2TaskBuffer deviceTasks(taskLimits);
    deviceTasks.upload(hostTasks);

    gl::gpu::Phase2FilterScheduleCapacity filterLimits{};
    filterLimits.calls = 4;
    filterLimits.examinedRows = 64;
    filterLimits.retainedRows = 64;
    filterLimits.maximumExaminedRowsPerCall = 64;
    gl::gpu::Phase2FilterScheduleArena filterSchedule(filterLimits);
    filterSchedule.appendCall(
        0, gl::gpu::DeviceHashMemoryKind::overall,
        analyzer.parameters.maxIterationNumberVariable, 1,
        block.statementCount);
    gl::gpu::CudaPhase2FilterSortBuffer deviceFilter(filterLimits);
    ASSERT_EQ(deviceFilter.filterAndSort(
        deviceProjection, filterSchedule), 5u);

    gl::gpu::Phase2GrowthScheduleArena growthSchedule(4);
    const gl::gpu::DevicePhase2GrowthCall scheduled =
        growthSchedule.appendCall(0, 0, 0);
    ASSERT_EQ(scheduled.taskIndex, 0u);
    ASSERT_EQ(scheduled.batchIndex, 0u);
    ASSERT_EQ(scheduled.filterCallIndex, 0u);
    growthSchedule.clear();
    ASSERT_TRUE(growthSchedule.calls.empty());
    growthSchedule.appendCall(0, 0, 0);

    gl::gpu::Phase2GrowthCapacity growthLimits{};
    growthLimits.calls = 4;
    growthLimits.retainedRows = 64;
    growthLimits.frontierRecords = 64;
    growthLimits.acceptedEvents = 128;
    growthLimits.rawRequests = 64;
    growthLimits.prefixPayloadValues = 512;
    growthLimits.prefixVariableValues = 256;
    growthLimits.prefixSecondaryValues = 256;
    growthLimits.candidateWindowRecords = 4;
    gl::gpu::CudaPhase2GrowthBuffer growth(growthLimits);
    const gl::gpu::Phase2GrowthResult result = growth.runRequestGrowth(
        deviceProjection, deviceTasks, deviceFilter, growthSchedule,
        gl::gpu::DevicePhase2GrowthParameters{
            analyzer.parameters.maxLenHypoKey,
            analyzer.parameters.maxNumberSecondaryVariables,
            analyzer.parameters.maxNumberSecondaryVariablesOrint,
            4,
            0 });
    if (result.acceptedEventCount != 7u) {
        std::fprintf(stderr, "[phase2-gpu] acceptedEventCount=%u\n",
                     result.acceptedEventCount);
        std::vector<gl::gpu::DeviceAcceptedGrowthEvent> diagnosticEvents(
            result.acceptedEventCount);
        growth.downloadAcceptedEvents(
            diagnosticEvents.data(), result.acceptedEventCount);
        for (const auto& event : diagnosticEvents) {
            std::fprintf(stderr, "[phase2-gpu] event count=%u flags=%u indices=",
                         event.count, event.flags);
            for (gl::NameId index = 0; index < event.count; ++index)
                std::fprintf(stderr, "%s%u", index == 0 ? "" : ",",
                             event.statementIndices[index]);
            std::fprintf(stderr, "\n");
        }
    }
    ASSERT_EQ(result.acceptedEventCount, 7u);
    ASSERT_EQ(result.rawRequestCount, 4u);
    ASSERT_TRUE(result.maximumFrontierCount > 0);
    ASSERT_TRUE(result.maximumCooperativeNodeCount > 0);
    ASSERT_TRUE(result.maximumPrefixPayloadValues > 0);
    ASSERT_TRUE(result.maximumPrefixVariableValues > 0);
    ASSERT_TRUE(result.maximumPrefixSecondaryValues > 0);
    uint32_t taskSubkeyCount = 0;
    ASSERT_EQ(growth.downloadTaskSubkeyCounts(&taskSubkeyCount, 1), 1u);
    ASSERT_EQ(taskSubkeyCount, 5u);

    std::vector<gl::gpu::DeviceAcceptedGrowthEvent> events(
        result.acceptedEventCount);
    ASSERT_EQ(growth.downloadAcceptedEvents(
        events.data(), static_cast<uint32_t>(events.size())),
        result.acceptedEventCount);
    const auto eventText = [](const auto& event) {
        std::string text = std::to_string(event.count) + ":";
        for (gl::NameId index = 0; index < event.count; ++index) {
            if (index > 0) text += ",";
            text += std::to_string(event.statementIndices[index]);
        }
        text += ":" + std::to_string(event.flags);
        return text;
    };
    std::vector<std::string> actualEvents;
    for (const auto& event : events) actualEvents.push_back(eventText(event));
    std::sort(actualEvents.begin(), actualEvents.end());
    std::vector<std::string> expectedEvents = {
        "1:0:1", "1:1:1", "1:2:7", "2:0,1:1",
        "2:0,2:6", "2:1,2:7", "3:0,1,2:6"
    };
    std::sort(expectedEvents.begin(), expectedEvents.end());
    ASSERT_EQ(actualEvents, expectedEvents);

    std::vector<gl::gpu::DeviceRawGrowthRequest> requests(
        result.rawRequestCount);
    ASSERT_EQ(growth.downloadRawRequests(
        requests.data(), static_cast<uint32_t>(requests.size())),
        result.rawRequestCount);
    std::vector<std::string> actualRequests;
    for (const auto& request : requests) {
        ASSERT_EQ(request.growthPosition, 0ull);
        actualRequests.push_back(eventText(request.event));
    }
    std::sort(actualRequests.begin(), actualRequests.end());
    std::vector<std::string> expectedRequests = {
        "1:2:7", "2:0,2:6", "2:1,2:7", "3:0,1,2:6"
    };
    std::sort(expectedRequests.begin(), expectedRequests.end());
    ASSERT_EQ(actualRequests, expectedRequests);

    {
        gl::gpu::CudaPhase2GrowthBuffer observationGrowth(growthLimits);
        const gl::gpu::Phase2GrowthResult observation =
            observationGrowth.runRequestGrowth(
                deviceProjection, deviceTasks, deviceFilter, growthSchedule,
                gl::gpu::DevicePhase2GrowthParameters{
                    analyzer.parameters.maxLenHypoKey,
                    analyzer.parameters.maxNumberSecondaryVariables,
                    analyzer.parameters.maxNumberSecondaryVariablesOrint,
                    4,
                    2 });
        ASSERT_EQ(observation.acceptedEventCount, result.acceptedEventCount);
        ASSERT_EQ(observation.rawRequestCount, result.rawRequestCount);
        uint64_t spanNodeTotal = 0;
        uint64_t spanCandidateTotal = 0;
        for (uint32_t bucket = 0;
             bucket < gl::gpu::kDeviceGrowthSpanBucketCount; ++bucket) {
            spanNodeTotal += observation.spanNodeCounts[bucket];
            spanCandidateTotal += observation.spanCandidateCounts[bucket];
            if (bucket != 2 && bucket != 3) {
                ASSERT_EQ(observation.spanNodeCounts[bucket], 0ull);
                ASSERT_EQ(observation.spanCandidateCounts[bucket], 0ull);
            }
        }
        ASSERT_EQ(observation.spanNodeCounts[2], 4ull);
        ASSERT_EQ(observation.spanCandidateCounts[2], 10ull);
        ASSERT_EQ(observation.spanNodeCounts[3], 2ull);
        ASSERT_EQ(observation.spanCandidateCounts[3], 9ull);
        ASSERT_EQ(spanNodeTotal, 6ull);
        ASSERT_EQ(spanCandidateTotal, 19ull);
        uint64_t gateCandidateTotal = 0;
        for (uint32_t depth = 1;
             depth < gl::gpu::kDeviceGrowthCensusDepthCount; ++depth) {
            const gl::gpu::Phase2GrowthGateCensus& census =
                observation.gateDepthCounts[depth];
            ASSERT_TRUE(census.candidateAttempts >= census.mandatoryReachable);
            ASSERT_TRUE(census.mandatoryReachable >= census.validityComparable);
            ASSERT_TRUE(census.validityComparable
                >= census.hypothesisCompatible);
            ASSERT_TRUE(census.hypothesisCompatible
                >= census.secondaryCompatible);
            ASSERT_TRUE(census.secondaryCompatible >= census.keyLengthAllowed);
            ASSERT_TRUE(census.keyLengthAllowed >= census.subkeyPresent);
            ASSERT_TRUE(census.subkeyPresent >= census.ownerSatisfied);
            ASSERT_TRUE(census.keyLengthAllowed >= census.wholeKeyPresent);
            ASSERT_TRUE(census.keyLengthAllowed >= census.termsSatisfied);
            ASSERT_TRUE(census.acceptedEvents
                <= census.ownerSatisfied + census.wholeKeyPresent);
            ASSERT_TRUE(census.ownerSatisfied >= census.children);
            gateCandidateTotal += census.candidateAttempts;
        }
        ASSERT_EQ(gateCandidateTotal, spanCandidateTotal);
    }

    gl::gpu::Phase2OrderingCapacity orderingLimits{};
    orderingLimits.events = 128;
    orderingLimits.deduplicationSlots = 128;
    orderingLimits.requests = 64;
    gl::gpu::CudaPhase2OrderingBuffer ordering(orderingLimits);
    const gl::gpu::Phase2OrderingResult ordered =
        ordering.orderAndDeduplicate(
            deviceProjection, deviceTasks, growth);
    ASSERT_EQ(ordered.orderedEventCount, 7u);
    ASSERT_EQ(ordered.uniqueRequestCount, 4u);
    std::vector<uint32_t> orderedIndices(ordered.orderedEventCount);
    ordering.downloadOrderedEventIndices(
        orderedIndices.data(), static_cast<uint32_t>(orderedIndices.size()));
    std::vector<std::string> orderedEventText;
    for (uint32_t eventIndex : orderedIndices)
        orderedEventText.push_back(eventText(events[eventIndex]));
    const std::vector<std::string> expectedOrderedEvents = {
        "1:0:1", "1:1:1", "1:2:7", "2:1,2:7",
        "2:0,1:1", "2:0,2:6", "3:0,1,2:6"
    };
    ASSERT_EQ(orderedEventText, expectedOrderedEvents);
    std::vector<gl::gpu::DeviceOrderedRequestToken> orderedRequests(
        ordered.uniqueRequestCount);
    ordering.downloadOrderedRequests(
        orderedRequests.data(),
        static_cast<uint32_t>(orderedRequests.size()));
    const uint64_t expectedGrowthPositions[4] = { 3, 4, 5, 5 };
    const uint32_t expectedEventOrders[4] = { 2, 3, 5, 6 };
    const std::string expectedOrderedRequests[4] = {
        "1:2:7", "2:1,2:7", "2:0,2:6", "3:0,1,2:6"
    };
    for (uint32_t index = 0; index < ordered.uniqueRequestCount; ++index) {
        ASSERT_EQ(orderedRequests[index].growthPosition,
                  expectedGrowthPositions[index]);
        ASSERT_EQ(orderedRequests[index].eventOrder,
                  expectedEventOrders[index]);
        ASSERT_EQ(eventText(events[orderedRequests[index].eventIndex]),
                  expectedOrderedRequests[index]);
    }

    gl::gpu::Phase2EvaluationCapacity evaluationLimits{};
    evaluationLimits.logicalBlocks = 1;
    evaluationLimits.requests = 8;
    evaluationLimits.reverseOwners = 16;
    evaluationLimits.candidateOwners = 8;
    evaluationLimits.encodedHits = 8;
    evaluationLimits.localValues = 8;
    evaluationLimits.firingRecords = 8;
    evaluationLimits.generatedBytes = 512;
    evaluationLimits.levelValues = 16;
    evaluationLimits.originDependencies = 16;
    evaluationLimits.markerKeys = 8;
    evaluationLimits.markerRemainingArgs = 8;
    evaluationLimits.markerArgs = 8;
    gl::gpu::CudaPhase2EvaluationBuffer evaluation(evaluationLimits);
    const gl::gpu::Phase2EvaluationResult dependencyRejected =
        evaluation.expandEvaluationWork(
            deviceProjection, deviceTasks, growth, ordering);
    ASSERT_EQ(dependencyRejected.requestCount, 4u);
    ASSERT_EQ(dependencyRejected.dependencyPassCount, 1u);
    ASSERT_EQ(dependencyRejected.reverseOwnerCount, 2u);
    ASSERT_EQ(dependencyRejected.candidateOwnerCount, 1u);
    ASSERT_EQ(dependencyRejected.encodedHitCount, 1u);
    ASSERT_EQ(dependencyRejected.localValueCount, 1u);

    for (int32_t statementIndex = 0;
         statementIndex < body.intEncodedStatements.size(); ++statementIndex) {
        const gl::IntEncodedExpr& expression =
            body.intEncodedStatements[statementIndex];
        body.intKnownStatements.upsert(
            gl::StatementKey{ expression.originalId, expression.validityId },
            gl::StatementFlags{ true, false });
        if (statementIndex != 2) {
            const int level = statementIndex == 0
                ? 2 : (statementIndex == 1 ? 3 : 7);
            body.intStatementLevelsMap.assignSet(
                gl::StatementKey{
                    expression.originalId, expression.validityId },
                &level, 1);
        }
    }
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer);
    deviceProjection.upload(hostProjection);
    const gl::gpu::Phase2EvaluationResult evaluationResult =
        evaluation.expandEvaluationWork(
            deviceProjection, deviceTasks, growth, ordering);
    ASSERT_EQ(evaluationResult.requestCount, 4u);
    ASSERT_EQ(evaluationResult.dependencyPassCount, 4u);
    ASSERT_EQ(evaluationResult.reverseOwnerCount, 5u);
    ASSERT_EQ(evaluationResult.candidateOwnerCount, 4u);
    ASSERT_EQ(evaluationResult.encodedHitCount, 4u);
    ASSERT_EQ(evaluationResult.localValueCount, 5u);
    std::vector<gl::gpu::DeviceEvaluationCandidate> evaluationCandidates(4);
    ASSERT_EQ(evaluation.downloadCandidates(
        evaluationCandidates.data(), 4), 4u);
    std::vector<uint32_t> candidateRequests;
    for (const auto& candidate : evaluationCandidates) {
        ASSERT_TRUE(candidate.encodedEntryIndex >= 0);
        candidateRequests.push_back(candidate.requestIndex);
    }
    std::sort(candidateRequests.begin(), candidateRequests.end());
    ASSERT_EQ(candidateRequests, std::vector<uint32_t>({ 0, 1, 2, 3 }));
    std::vector<gl::gpu::DeviceEvaluationCandidate> encodedHits(4);
    ASSERT_EQ(evaluation.downloadEncodedHits(encodedHits.data(), 4), 4u);
    std::vector<uint32_t> hitRequests;
    for (const auto& hit : encodedHits)
        hitRequests.push_back(hit.requestIndex);
    std::sort(hitRequests.begin(), hitRequests.end());
    ASSERT_EQ(hitRequests, std::vector<uint32_t>({ 0, 1, 2, 3 }));
    std::vector<gl::gpu::DeviceEvaluationValueWork> valueWork(5);
    ASSERT_EQ(evaluation.downloadValueWork(valueWork.data(), 5), 5u);
    std::vector<uint32_t> valueRequests;
    for (const auto& work : valueWork)
        valueRequests.push_back(work.requestIndex);
    std::sort(valueRequests.begin(), valueRequests.end());
    ASSERT_EQ(valueRequests, std::vector<uint32_t>({ 0, 1, 2, 2, 3 }));

    const gl::gpu::Phase2FiringExpressionResult firingExpressions =
        evaluation.materializeFiringExpressions(
            deviceProjection, deviceTasks, growth, ordering);
    if (firingExpressions.firingRecordCount != 5u) {
        std::fprintf(stderr,
            "[phase2-gpu] firing records=%u bytes=%u levels=%u origins=%u "
            "markerKeys=%u markerRemaining=%u markerArgs=%u\n",
            firingExpressions.firingRecordCount,
            firingExpressions.generatedByteCount,
            firingExpressions.levelValueCount,
            firingExpressions.originDependencyCount,
            firingExpressions.markerKeyCount,
            firingExpressions.markerRemainingArgCount,
            firingExpressions.markerArgCount);
    }
    ASSERT_EQ(firingExpressions.firingRecordCount, 5u);
    ASSERT_TRUE(firingExpressions.generatedByteCount > 0);
    ASSERT_EQ(firingExpressions.levelValueCount, 12u);
    ASSERT_EQ(firingExpressions.originDependencyCount, 12u);
    ASSERT_EQ(firingExpressions.markerKeyCount, 2u);
    ASSERT_EQ(firingExpressions.markerRemainingArgCount, 2u);
    ASSERT_EQ(firingExpressions.markerArgCount, 1u);
    std::vector<gl::gpu::DevicePhase2FiringRecord> firingRecords(5);
    ASSERT_EQ(evaluation.downloadFiringRecords(
        firingRecords.data(), static_cast<uint32_t>(firingRecords.size())), 5u);
    std::vector<char> generatedBytes(firingExpressions.generatedByteCount);
    ASSERT_EQ(evaluation.downloadGeneratedBytes(
        generatedBytes.data(), static_cast<uint32_t>(generatedBytes.size())),
        firingExpressions.generatedByteCount);
    std::vector<int32_t> firingLevels(firingExpressions.levelValueCount);
    ASSERT_EQ(evaluation.downloadLevelValues(
        firingLevels.data(), static_cast<uint32_t>(firingLevels.size())), 12u);
    std::vector<gl::gpu::DeviceEvaluationDependency> firingOrigins(
        firingExpressions.originDependencyCount);
    ASSERT_EQ(evaluation.downloadOriginDependencies(
        firingOrigins.data(), static_cast<uint32_t>(firingOrigins.size())), 12u);
    std::vector<gl::gpu::DeviceEvaluationByteSlice> markerKeys(
        firingExpressions.markerKeyCount);
    ASSERT_EQ(evaluation.downloadMarkerKeys(
        markerKeys.data(), static_cast<uint32_t>(markerKeys.size())), 2u);
    std::vector<int32_t> markerRemaining(
        firingExpressions.markerRemainingArgCount);
    ASSERT_EQ(evaluation.downloadMarkerRemainingArgs(
        markerRemaining.data(),
        static_cast<uint32_t>(markerRemaining.size())), 2u);
    std::vector<gl::gpu::DeviceEvaluationByteSlice> markerArgs(
        firingExpressions.markerArgCount);
    ASSERT_EQ(evaluation.downloadMarkerArgs(
        markerArgs.data(), static_cast<uint32_t>(markerArgs.size())), 1u);
    ASSERT_EQ(evaluation.orderFiringRecords(deviceProjection), 5u);
    std::vector<uint32_t> firingOrder(5);
    ASSERT_EQ(evaluation.downloadFiringOrder(
        firingOrder.data(), static_cast<uint32_t>(firingOrder.size())), 5u);
    const uint32_t expectedFiringRequestOrder[5] = { 2, 3, 0, 1, 2 };
    const uint64_t expectedFiringGrowthOrder[5] = { 5, 5, 3, 4, 5 };
    for (uint32_t index = 0; index < firingOrder.size(); ++index) {
        ASSERT_TRUE(firingOrder[index] < firingRecords.size());
        const gl::gpu::DevicePhase2FiringRecord& orderedRecord =
            firingRecords[firingOrder[index]];
        ASSERT_EQ(orderedRecord.requestIndex, expectedFiringRequestOrder[index]);
        ASSERT_EQ(orderedRecord.growthPosition, expectedFiringGrowthOrder[index]);
        ASSERT_EQ(orderedRecord.logicalBlockIndex, 0u);
        ASSERT_EQ(orderedRecord.partOrdinal, 0u);
        ASSERT_EQ((orderedRecord.flags & gl::gpu::deviceFiringMarker) != 0,
            index == 4);
    }
    ASSERT_EQ(evaluation.selectDoomPrefixes(deviceProjection), 5u);
    int64_t noDoomLine = 0;
    ASSERT_EQ(evaluation.downloadDoomLines(&noDoomLine, 1), 1u);
    ASSERT_EQ(noDoomLine, std::numeric_limits<int64_t>::max());
    uint32_t noDoomRequest = 0;
    ASSERT_EQ(evaluation.downloadDoomRequestIndices(&noDoomRequest, 1), 1u);
    ASSERT_EQ(noDoomRequest, std::numeric_limits<uint32_t>::max());
    const auto sliceText = [&](const gl::gpu::DeviceEvaluationByteSlice& slice) {
        ASSERT_TRUE(slice.offset <= generatedBytes.size());
        ASSERT_TRUE(slice.length <= generatedBytes.size() - slice.offset);
        return std::string(
            generatedBytes.data() + slice.offset,
            static_cast<std::size_t>(slice.length));
    };

    gl::gpu::Phase2SealingArena sealing(evaluationLimits);
    gl::SealedPageSet sealedOutput;
    sealedOutput.bind(&gl::staticMemory());
    gl::SealedPageSet* sealedOutputs[1] = { &sealedOutput };
    ASSERT_EQ(sealing.downloadAndSeal(
        evaluation, hostProjection, firingExpressions, 5,
        sealedOutputs, 1), 5u);
    ASSERT_TRUE(sealedOutput.sealed());
    ASSERT_EQ(sealedOutput.recordCount(), 5);
    uint32_t sealedIndex = 0;
    sealedOutput.forEachRecord<gl::FiringRecord>(
        [&](const gl::FiringRecord& sealed) {
            ASSERT_TRUE(sealedIndex < firingOrder.size());
            const gl::gpu::DevicePhase2FiringRecord& source =
                firingRecords[firingOrder[sealedIndex]];
            ASSERT_EQ(sealed.rplExpr2.toStdString(), sliceText(source.expression));
            ASSERT_EQ(sealed.validityName.toStdString(),
                body.nameMap.decode(source.validityId));
            const bool sourceMarker = (source.flags
                & gl::gpu::deviceFiringMarker) != 0;
            ASSERT_EQ(sealed.isMarker, sourceMarker);
            ASSERT_FALSE(sealed.isOrdis2Demand);
            if (!sourceMarker) {
                ASSERT_EQ(sealed.originTag.toStdString(), "implication");
                ASSERT_EQ(sealed.levels.size(),
                    static_cast<int32_t>(source.levelsCount));
                for (uint32_t level = 0; level < source.levelsCount; ++level) {
                    ASSERT_EQ(sealed.levels[static_cast<int32_t>(level)],
                        firingLevels[source.levelsOffset + level]);
                }
                ASSERT_EQ(sealed.originDeps.size(),
                    static_cast<int32_t>(source.originDependencyCount));
                for (uint32_t dependency = 0;
                     dependency < source.originDependencyCount; ++dependency) {
                    const gl::gpu::DeviceEvaluationDependency& expected =
                        firingOrigins[source.originDependencyOffset + dependency];
                    const std::string expectedOriginal = dependency == 0
                        ? body.ruleInterner.decode(
                            static_cast<int32_t>(expected.originalId))
                        : body.nameMap.decode(expected.originalId);
                    ASSERT_EQ(sealed.originDeps[
                        static_cast<int32_t>(dependency)].original.toStdString(),
                        expectedOriginal);
                    ASSERT_EQ(sealed.originDeps[
                        static_cast<int32_t>(dependency)].validityName.toStdString(),
                        body.nameMap.decode(expected.validityId));
                }
                ASSERT_EQ(sealed.doNotDisintegrate,
                    (source.flags
                        & gl::gpu::deviceFiringDoNotDisintegrate) != 0);
                ASSERT_EQ(sealed.allowOrDisintegration,
                    (source.flags
                        & gl::gpu::deviceFiringAllowOrDisintegration) != 0);
                ASSERT_EQ(sealed.allGood,
                    (source.flags & gl::gpu::deviceFiringAllGood) != 0);
                ASSERT_EQ(sealed.alreadyKnown,
                    (source.flags & gl::gpu::deviceFiringAlreadyKnown) != 0);
                ASSERT_EQ(sealed.iteration, source.iteration);
            }
            else {
                ASSERT_EQ(sealed.markerArgsSorted.size(),
                    static_cast<int32_t>(source.markerArgCount));
                for (uint32_t argument = 0;
                     argument < source.markerArgCount; ++argument) {
                    ASSERT_EQ(sealed.markerArgsSorted[
                        static_cast<int32_t>(argument)].toStdString(),
                        sliceText(markerArgs[source.markerArgOffset + argument]));
                }
                ASSERT_EQ(sealed.admv.key.size(),
                    static_cast<int32_t>(source.markerKeyCount));
                for (uint32_t key = 0; key < source.markerKeyCount; ++key) {
                    ASSERT_EQ(sealed.admv.key[
                        static_cast<int32_t>(key)].toStdString(),
                        sliceText(markerKeys[source.markerKeyOffset + key]));
                }
                ASSERT_EQ(sealed.admv.remainingArgsSorted.size(),
                    static_cast<int32_t>(source.markerRemainingArgCount));
                for (uint32_t argument = 0;
                     argument < source.markerRemainingArgCount; ++argument) {
                    ASSERT_EQ(sealed.admv.remainingArgsSorted[
                        static_cast<int32_t>(argument)].toStdString(),
                        body.ruleInterner.decode(markerRemaining[
                            source.markerRemainingArgOffset + argument]));
                }
                ASSERT_EQ(sealed.admv.standardMaxAdmissionDepth,
                    source.standardMaxAdmissionDepth);
                ASSERT_EQ(sealed.admv.standardMaxSecondaryNumber,
                    source.standardMaxSecondaryNumber);
                ASSERT_FALSE(sealed.admv.flag);
                ASSERT_EQ(sealed.admv.ordisOnly,
                    (source.flags & gl::gpu::deviceFiringOrdisOnly) != 0);
                ASSERT_EQ(sealed.markerNotAtomic,
                    (source.flags
                        & gl::gpu::deviceFiringMarkerNotAtomic) != 0);
            }
            ++sealedIndex;
        });
    ASSERT_EQ(sealedIndex, 5u);
    sealedOutput.freePages();

    const std::vector<std::vector<int32_t>> expectedLevelsByRequest = {
        { 4, 5 }, { 3, 4, 5 }, { 2, 4, 5 }, { 2, 3, 4, 5 }
    };
    const std::vector<std::vector<int32_t>> expectedPremisesByRequest = {
        { 2 }, { 1, 2 }, { 0, 2 }, { 0, 1, 2 }
    };
    std::vector<std::string> firedExpressions;
    std::vector<uint32_t> firedRequests;
    uint32_t markerRecordCount = 0;
    for (const auto& record : firingRecords) {
        const std::string expression = sliceText(record.expression);
        if ((record.flags & gl::gpu::deviceFiringMarker) != 0) {
            ++markerRecordCount;
            ASSERT_EQ(record.requestIndex, 2u);
            ASSERT_EQ(expression,
                std::string("(gpuMarker[marker,product_variable])"));
            ASSERT_EQ(record.validityId, markerOrBranchId);
            ASSERT_EQ(record.iteration, -1);
            ASSERT_EQ(record.demandSourceImplId, 0);
            ASSERT_EQ(record.standardMaxAdmissionDepth, 3);
            ASSERT_EQ(record.standardMaxSecondaryNumber, 9);
            ASSERT_EQ(record.flags, static_cast<uint32_t>(
                gl::gpu::deviceFiringMarker
                | gl::gpu::deviceFiringDisintegrationAllowed
                | gl::gpu::deviceFiringOrdisOnly
                | gl::gpu::deviceFiringMarkerNotAtomic));
            ASSERT_EQ(record.levelsCount, 0u);
            ASSERT_EQ(record.originDependencyCount, 0u);
            ASSERT_EQ(record.markerKeyCount, 2u);
            ASSERT_EQ(record.markerRemainingArgCount, 2u);
            ASSERT_EQ(record.markerArgCount, 1u);
            ASSERT_EQ(sliceText(markerKeys[record.markerKeyOffset]),
                std::string("(gpuKeyA[product_variable,tail])"));
            ASSERT_EQ(sliceText(markerKeys[record.markerKeyOffset + 1]),
                std::string("(gpuKeyB[static,product_variable])"));
            ASSERT_EQ(markerRemaining[record.markerRemainingArgOffset],
                gpuRemainingAlphaId);
            ASSERT_EQ(markerRemaining[
                record.markerRemainingArgOffset + 1], gpuRemainingZetaId);
            ASSERT_EQ(sliceText(markerArgs[record.markerArgOffset]),
                std::string("product_variable"));
            continue;
        }
        firedExpressions.push_back(expression);
        firedRequests.push_back(record.requestIndex);
        ASSERT_EQ(record.validityId,
            record.requestIndex == 0 || record.requestIndex == 2
                ? mainId : childId);
        ASSERT_EQ(record.iteration, 0);
        ASSERT_EQ(record.demandSourceImplId, 0);
        ASSERT_EQ(record.standardMaxAdmissionDepth, 0);
        ASSERT_EQ(record.standardMaxSecondaryNumber, 0);
        uint32_t expectedFlags =
            gl::gpu::deviceFiringDisintegrationAllowed
            | gl::gpu::deviceFiringAlreadyKnown;
        if (record.requestIndex == 0 || record.requestIndex == 2)
            expectedFlags |= gl::gpu::deviceFiringAllGood;
        if (record.flags != expectedFlags) {
            std::fprintf(stderr,
                "[phase2-gpu] head request=%u flags=%u expected=%u expression=%s\n",
                record.requestIndex, record.flags, expectedFlags,
                expression.c_str());
        }
        ASSERT_EQ(record.flags, expectedFlags);
        ASSERT_TRUE(record.requestIndex < expectedLevelsByRequest.size());
        ASSERT_TRUE(record.levelsOffset <= firingLevels.size());
        ASSERT_TRUE(record.levelsCount
            <= firingLevels.size() - record.levelsOffset);
        const std::vector<int32_t> actualLevels(
            firingLevels.begin() + record.levelsOffset,
            firingLevels.begin() + record.levelsOffset + record.levelsCount);
        ASSERT_EQ(actualLevels, expectedLevelsByRequest[record.requestIndex]);
        const std::vector<int32_t>& expectedPremises =
            expectedPremisesByRequest[record.requestIndex];
        ASSERT_EQ(record.originDependencyCount,
            static_cast<uint32_t>(expectedPremises.size() + 1));
        ASSERT_TRUE(record.originDependencyOffset <= firingOrigins.size());
        ASSERT_TRUE(record.originDependencyCount
            <= firingOrigins.size() - record.originDependencyOffset);
        const auto& sourceDependency = firingOrigins[
            record.originDependencyOffset];
        ASSERT_EQ(sourceDependency.originalId, gpuSourceId);
        ASSERT_EQ(sourceDependency.validityId, mainId);
        for (uint32_t dependency = 0;
             dependency < expectedPremises.size(); ++dependency) {
            const gl::IntEncodedExpr& expectedPremise =
                body.intEncodedStatements[
                    expectedPremises[dependency]];
            const auto& actualDependency = firingOrigins[
                record.originDependencyOffset + dependency + 1];
            ASSERT_EQ(actualDependency.originalId, expectedPremise.originalId);
            ASSERT_EQ(actualDependency.validityId, expectedPremise.validityId);
        }
        ASSERT_EQ(record.markerKeyCount, 0u);
        ASSERT_EQ(record.markerRemainingArgCount, 0u);
        ASSERT_EQ(record.markerArgCount, 0u);
    }
    ASSERT_EQ(markerRecordCount, 1u);
    std::sort(firedExpressions.begin(), firedExpressions.end());
    std::vector<std::string> expectedFiredExpressions = {
        "(gpuHead[secondary_v_shadow,static])",
        "(gpuHead[secondary_v_shadow,static])",
        "(gpuHead[product_variable,static])",
        "(gpuHead[product_variable,static])"
    };
    std::sort(expectedFiredExpressions.begin(), expectedFiredExpressions.end());
    ASSERT_EQ(firedExpressions, expectedFiredExpressions);

    body.intToBeProved.assignSet(
        gl::packStatementKey(knownSecondaryHead, mainId), nullptr, 0);
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer);
    deviceProjection.upload(hostProjection);
    ASSERT_EQ(evaluation.orderFiringRecords(deviceProjection), 5u);
    const uint32_t goalRetainedCount =
        evaluation.selectDoomPrefixes(deviceProjection);
    int64_t goalDoomLine = 0;
    ASSERT_EQ(evaluation.downloadDoomLines(&goalDoomLine, 1), 1u);
    if (goalRetainedCount != 1u) {
        std::fprintf(stderr,
            "[phase2-gpu] goal retained=%u doom=%lld\n",
            goalRetainedCount, static_cast<long long>(goalDoomLine));
        for (const auto& record : firingRecords) {
            std::fprintf(stderr,
                "[phase2-gpu] firing request=%u growth=%llu expression=%s flags=%u\n",
                record.requestIndex,
                static_cast<unsigned long long>(record.growthPosition),
                sliceText(record.expression).c_str(), record.flags);
        }
    }
    ASSERT_EQ(goalRetainedCount, 1u);
    ASSERT_EQ(goalDoomLine, static_cast<int64_t>(3ull << 14));
    uint32_t goalDoomRequest = 5;
    ASSERT_EQ(evaluation.downloadDoomRequestIndices(
        &goalDoomRequest, 1), 1u);
    ASSERT_EQ(goalDoomRequest, 0u);
    uint32_t retainedGoalRecord = 5;
    ASSERT_EQ(evaluation.downloadFiringOrder(&retainedGoalRecord, 1), 1u);
    ASSERT_TRUE(retainedGoalRecord < firingRecords.size());
    ASSERT_EQ(firingRecords[retainedGoalRecord].requestIndex, 0u);
    ASSERT_EQ(firingRecords[retainedGoalRecord].growthPosition, 3ull);

    body.intToBeProved.resetToFresh();
    body.intToBeProved.assignSet(
        gl::packStatementKey(knownProductHead, mainId), nullptr, 0);
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer);
    deviceProjection.upload(hostProjection);
    ASSERT_EQ(evaluation.orderFiringRecords(deviceProjection), 5u);
    ASSERT_EQ(evaluation.selectDoomPrefixes(deviceProjection), 4u);
    int64_t sharedPositionDoomLine = 0;
    ASSERT_EQ(evaluation.downloadDoomLines(
        &sharedPositionDoomLine, 1), 1u);
    ASSERT_EQ(sharedPositionDoomLine, static_cast<int64_t>(5ull << 14));
    uint32_t sharedPositionDoomRequest = 5;
    ASSERT_EQ(evaluation.downloadDoomRequestIndices(
        &sharedPositionDoomRequest, 1), 1u);
    ASSERT_EQ(sharedPositionDoomRequest, 2u);
    uint32_t sharedPositionOrder[4]{};
    ASSERT_EQ(evaluation.downloadFiringOrder(sharedPositionOrder, 4), 4u);
    const uint32_t expectedSharedPositionRequests[4] = { 2, 0, 1, 2 };
    for (uint32_t index = 0; index < 4; ++index) {
        ASSERT_EQ(firingRecords[sharedPositionOrder[index]].requestIndex,
            expectedSharedPositionRequests[index]);
    }

    analyzer.parameters.compressor_mode = true;
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer);
    deviceProjection.upload(hostProjection);
    ASSERT_EQ(evaluation.orderFiringRecords(deviceProjection), 5u);
    ASSERT_EQ(evaluation.selectDoomPrefixes(deviceProjection), 5u);
    int64_t compressorDoomLine = 0;
    ASSERT_EQ(evaluation.downloadDoomLines(&compressorDoomLine, 1), 1u);
    ASSERT_EQ(compressorDoomLine, std::numeric_limits<int64_t>::max());
    uint32_t compressorDoomRequest = 0;
    ASSERT_EQ(evaluation.downloadDoomRequestIndices(
        &compressorDoomRequest, 1), 1u);
    ASSERT_EQ(compressorDoomRequest, std::numeric_limits<uint32_t>::max());

    analyzer.parameters.compressor_mode = false;
    body.intToBeProved.resetToFresh();
    body.primedForContradiction = true;
    body.intKnownStatements.upsert(
        gl::StatementKey{ negatedSecondaryHead, mainId },
        gl::StatementFlags{ true, false });
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer);
    deviceProjection.upload(hostProjection);
    ASSERT_EQ(evaluation.orderFiringRecords(deviceProjection), 5u);
    ASSERT_EQ(evaluation.selectDoomPrefixes(deviceProjection), 1u);
    int64_t contradictionDoomLine = 0;
    ASSERT_EQ(evaluation.downloadDoomLines(&contradictionDoomLine, 1), 1u);
    ASSERT_EQ(contradictionDoomLine, static_cast<int64_t>(3ull << 14));
    uint32_t contradictionDoomRequest = 5;
    ASSERT_EQ(evaluation.downloadDoomRequestIndices(
        &contradictionDoomRequest, 1), 1u);
    ASSERT_EQ(contradictionDoomRequest, 0u);
    std::sort(firedRequests.begin(), firedRequests.end());
    ASSERT_EQ(firedRequests, std::vector<uint32_t>({ 0, 1, 2, 3 }));

    gl::ExpressionStump splitStumps[2]{};
    splitStumps[0].allIdx[0] = 2;
    splitStumps[0].count = 1;
    splitStumps[0].terminalOnly = 1;
    splitStumps[1].allIdx[0] = 0;
    splitStumps[1].allIdx[1] = 1;
    splitStumps[1].count = 2;
    hostTasks.clear();
    hostTasks.appendTask(
        0, &batch, 1, splitStumps, 2, 0, 1,
        analyzer.parameters.maxIterationNumberVariable, 0);
    deviceTasks.upload(hostTasks);
    const gl::gpu::Phase2GrowthResult stumpResult = growth.runRequestGrowth(
        deviceProjection, deviceTasks, deviceFilter, growthSchedule,
        gl::gpu::DevicePhase2GrowthParameters{
            analyzer.parameters.maxLenHypoKey,
            analyzer.parameters.maxNumberSecondaryVariables,
            analyzer.parameters.maxNumberSecondaryVariablesOrint,
            gl::gpu::kDeviceGrowthProductionCooperativeSpan,
            0 });
    ASSERT_EQ(stumpResult.acceptedEventCount, 3u);
    ASSERT_EQ(stumpResult.rawRequestCount, 2u);
    std::vector<gl::gpu::DeviceAcceptedGrowthEvent> stumpEvents(
        stumpResult.acceptedEventCount);
    growth.downloadAcceptedEvents(
        stumpEvents.data(), static_cast<uint32_t>(stumpEvents.size()));
    std::vector<std::string> actualStumpEvents;
    for (const auto& event : stumpEvents)
        actualStumpEvents.push_back(eventText(event));
    std::sort(actualStumpEvents.begin(), actualStumpEvents.end());
    std::vector<std::string> expectedStumpEvents = {
        "1:2:7", "2:0,1:1", "3:0,1,2:6"
    };
    std::sort(expectedStumpEvents.begin(), expectedStumpEvents.end());
    ASSERT_EQ(actualStumpEvents, expectedStumpEvents);
    const gl::gpu::Phase2OrderingResult orderedStumps =
        ordering.orderAndDeduplicate(
            deviceProjection, deviceTasks, growth);
    ASSERT_EQ(orderedStumps.orderedEventCount, 3u);
    ASSERT_EQ(orderedStumps.uniqueRequestCount, 2u);
    std::vector<uint32_t> orderedStumpIndices(
        orderedStumps.orderedEventCount);
    ordering.downloadOrderedEventIndices(
        orderedStumpIndices.data(),
        static_cast<uint32_t>(orderedStumpIndices.size()));
    ASSERT_EQ(eventText(stumpEvents[orderedStumpIndices[0]]), "1:2:7");
    ASSERT_EQ(eventText(stumpEvents[orderedStumpIndices[1]]), "2:0,1:1");
    ASSERT_EQ(eventText(stumpEvents[orderedStumpIndices[2]]), "3:0,1,2:6");
    std::vector<gl::gpu::DeviceOrderedRequestToken> orderedStumpRequests(
        orderedStumps.uniqueRequestCount);
    ordering.downloadOrderedRequests(
        orderedStumpRequests.data(),
        static_cast<uint32_t>(orderedStumpRequests.size()));
    ASSERT_EQ(orderedStumpRequests[0].growthPosition, 1ull);
    ASSERT_EQ(orderedStumpRequests[0].eventOrder, 0u);
    ASSERT_EQ(orderedStumpRequests[1].growthPosition, 2ull);
    ASSERT_EQ(orderedStumpRequests[1].eventOrder, 2u);
}

TEST(phase2_projection, cuda_ordering_deduplicates_first_request_per_call) {
    gl::ExpressionAnalyzer analyzer("Peano");
    analyzer.parameters.maxIterationNumberVariable = 3;
    gl::Memory body;
    body.setExprKey("cuda_ordering_dedup");
    gl::NameMap& names = body.nameMap;
    const gl::NameId mainId = names.encode("main");
    const gl::NameId nameId = names.encode("duplicate_name");
    const gl::NameId originalId = names.encode("(duplicate_name[x])");
    const gl::NameId argumentId = names.encode("duplicate_argument");
    const gl::IntEncodedExpr statement = projectionStatement(
        nameId, originalId, mainId, argumentId);
    body.intEncodedStatements.push_back(statement);
    body.intEncodedStatements.push_back(statement);
    const gl::IntEncodedExpr* expressions[1] = {
        &body.intEncodedStatements[0]
    };
    gl::NameId keyData[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
    const gl::NameId keyLength = analyzer.makeIntNormalizedKeyFromEncoded(
        expressions, 1, keyData, gl::ExecutionParameters::MAX_KEY_SLOTS);
    const gl::NormKey key{
        1, std::vector<gl::NameId>(keyData, keyData + keyLength) };
    gl::OwnerSet owner;
    owner.hasLooseOwner = true;
    const int32_t fixtureRuleId = body.ruleInterner.encode(
        "(cuda-ordering-rule[1])");
    const gl::RuleOwner fixtureOwner = gl::packRuleOwner(
        fixtureRuleId, mainId);
    owner.owners.emplace_back(fixtureOwner, -1);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    body.overallHashMemory.normalizedEncodedSubkeys.assignRun(
        key, std::vector<gl::OwnerSet>{ owner });
    gl::ExpressionAnalyzer::addWholeKeyOwner(
        body.overallHashMemory.normalizedEncodedKeys,
        key.numberExpressions, key.data.data(),
        static_cast<int32_t>(key.data.size()), fixtureOwner, ownerArena);
    body.overallHashMemory.maxKeyLength = 1;

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= names.nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(names.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity projectionLimits =
        projectionCapacity(1, 2,
            static_cast<uint32_t>(names.nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena hostProjection(projectionLimits);
    const gl::gpu::DeviceLogicalBlockProjection block =
        hostProjection.appendLogicalBlock(body, analyzer);
    gl::gpu::CudaPhase2ProjectionBuffer deviceProjection(projectionLimits);
    deviceProjection.upload(hostProjection);

    gl::gpu::Phase2RequestBatchInput batches[2]{};
    for (auto& batch : batches) {
        batch.kind = gl::gpu::DeviceRequestBatchKind::newThisBurst;
        batch.memory = gl::gpu::DeviceHashMemoryKind::overall;
    }
    gl::gpu::Phase2TaskProjectionCapacity taskLimits{};
    taskLimits.tasks = 2;
    taskLimits.batches = 4;
    taskLimits.terms = 2;
    taskLimits.stumps = 2;
    gl::gpu::Phase2TaskProjectionArena hostTasks(taskLimits);
    hostTasks.appendTask(
        0, batches, 2, nullptr, 0, 0, 0,
        analyzer.parameters.maxIterationNumberVariable, 0);
    gl::gpu::CudaPhase2TaskBuffer deviceTasks(taskLimits);
    deviceTasks.upload(hostTasks);

    gl::gpu::Phase2FilterScheduleCapacity filterLimits{};
    filterLimits.calls = 4;
    filterLimits.examinedRows = 16;
    filterLimits.retainedRows = 16;
    filterLimits.maximumExaminedRowsPerCall = 8;
    gl::gpu::Phase2FilterScheduleArena filterSchedule(filterLimits);
    for (uint32_t call = 0; call < 2; ++call) {
        filterSchedule.appendCall(
            0, gl::gpu::DeviceHashMemoryKind::overall,
            analyzer.parameters.maxIterationNumberVariable, 1,
            block.statementCount);
    }
    gl::gpu::CudaPhase2FilterSortBuffer deviceFilter(filterLimits);
    ASSERT_EQ(filterSchedule.classes.size(), 1u);
    ASSERT_EQ(filterSchedule.callClassIndices[0], 0u);
    ASSERT_EQ(filterSchedule.callClassIndices[1], 0u);
    ASSERT_EQ(deviceFilter.filterAndSort(
        deviceProjection, filterSchedule), 2u);
    uint32_t duplicateCallCounts[2]{};
    ASSERT_EQ(deviceFilter.downloadCallCounts(duplicateCallCounts, 2), 2u);
    ASSERT_EQ(duplicateCallCounts[0], 2u);
    ASSERT_EQ(duplicateCallCounts[1], 2u);

    gl::gpu::Phase2GrowthScheduleArena growthSchedule(4);
    growthSchedule.appendCall(0, 0, 0);
    growthSchedule.appendCall(0, 1, 1);
    gl::gpu::Phase2GrowthCapacity growthLimits{};
    growthLimits.calls = 4;
    growthLimits.retainedRows = 16;
    growthLimits.frontierRecords = 16;
    growthLimits.acceptedEvents = 16;
    growthLimits.rawRequests = 8;
    growthLimits.prefixPayloadValues = 256;
    growthLimits.prefixVariableValues = 128;
    growthLimits.prefixSecondaryValues = 128;
    growthLimits.candidateWindowRecords = 16;
    gl::gpu::CudaPhase2GrowthBuffer growth(growthLimits);
    const gl::gpu::Phase2GrowthResult growthResult = growth.runRequestGrowth(
        deviceProjection, deviceTasks, deviceFilter, growthSchedule,
        gl::gpu::DevicePhase2GrowthParameters{
            8, 8, 8, gl::gpu::kDeviceGrowthProductionCooperativeSpan, 0 });
    ASSERT_EQ(growthResult.acceptedEventCount, 4u);
    ASSERT_EQ(growthResult.rawRequestCount, 4u);

    gl::gpu::Phase2OrderingCapacity orderingLimits{};
    orderingLimits.events = 16;
    orderingLimits.deduplicationSlots = 16;
    orderingLimits.requests = 8;
    gl::gpu::CudaPhase2OrderingBuffer ordering(orderingLimits);
    const gl::gpu::Phase2OrderingResult result =
        ordering.orderAndDeduplicate(
            deviceProjection, deviceTasks, growth);
    ASSERT_EQ(result.orderedEventCount, 4u);
    ASSERT_EQ(result.uniqueRequestCount, 2u);
    std::vector<gl::gpu::DeviceAcceptedGrowthEvent> events(4);
    growth.downloadAcceptedEvents(events.data(), 4);
    std::vector<uint32_t> eventOrder(4);
    ordering.downloadOrderedEventIndices(eventOrder.data(), 4);
    ASSERT_EQ(events[eventOrder[0]].callIndex, 0u);
    ASSERT_EQ(events[eventOrder[0]].statementIndices[0], 0);
    ASSERT_EQ(events[eventOrder[1]].callIndex, 0u);
    ASSERT_EQ(events[eventOrder[1]].statementIndices[0], 1);
    ASSERT_EQ(events[eventOrder[2]].callIndex, 1u);
    ASSERT_EQ(events[eventOrder[2]].statementIndices[0], 0);
    ASSERT_EQ(events[eventOrder[3]].callIndex, 1u);
    ASSERT_EQ(events[eventOrder[3]].statementIndices[0], 1);
    std::vector<gl::gpu::DeviceOrderedRequestToken> requests(2);
    ordering.downloadOrderedRequests(requests.data(), 2);
    ASSERT_EQ(requests[0].eventOrder, 0u);
    ASSERT_EQ(requests[0].growthPosition, 1ull);
    ASSERT_EQ(events[requests[0].eventIndex].callIndex, 0u);
    ASSERT_EQ(events[requests[0].eventIndex].statementIndices[0], 0);
    ASSERT_EQ(requests[1].eventOrder, 2u);
    ASSERT_EQ(requests[1].growthPosition, 3ull);
    ASSERT_EQ(events[requests[1].eventIndex].callIndex, 1u);
    ASSERT_EQ(events[requests[1].eventIndex].statementIndices[0], 0);
}

TEST(phase2_projection, fixed_cuda_phase2_buffers_report_owned_bytes) {
    gl::gpu::Phase2ProjectionCapacity projectionCapacity{};
    projectionCapacity.logicalBlocks = 512;
    projectionCapacity.statements = 262144;
    projectionCapacity.nameRecords = 524288;
    projectionCapacity.nameBytes = 16777216;
    projectionCapacity.nameSlots = 1048576;
    projectionCapacity.ruleStringRecords = 131072;
    projectionCapacity.ruleStringBytes = 8388608;
    projectionCapacity.byteMapViews = 5120;
    projectionCapacity.byteMapEntries = 524288;
    projectionCapacity.byteMapSlots = 2097152;
    projectionCapacity.byteKeyBytes = 67108864;
    projectionCapacity.blobRecords = 524288;
    projectionCapacity.blobBytes = 33554432;
    projectionCapacity.reverseMapViews = 512;
    projectionCapacity.reverseMapEntries = 65536;
    projectionCapacity.reverseMapSlots = 262144;
    projectionCapacity.reverseKeyBytes = 16777216;
    projectionCapacity.reverseOwners = 65536;
    projectionCapacity.podMapViews = 5632;
    projectionCapacity.podMapEntries = 524288;
    projectionCapacity.podMapSlots = 2097152;
    projectionCapacity.podRunValues = 524288;
    projectionCapacity.mandatoryStatementKeys = 262144;
    projectionCapacity.metadataBytes = 16384;
    gl::gpu::CudaPhase2ProjectionBuffer projection(projectionCapacity);
    ASSERT_EQ(projection.fixedAllocationBytes(), 299951120ull);

    gl::gpu::Phase2TaskProjectionCapacity taskCapacity{};
    taskCapacity.tasks = 512;
    taskCapacity.batches = 2048;
    taskCapacity.terms = 2048;
    taskCapacity.stumps = 16384;
    gl::gpu::CudaPhase2TaskBuffer tasks(taskCapacity);
    ASSERT_EQ(tasks.fixedAllocationBytes(), 731144ull);

    gl::gpu::Phase2FilterScheduleCapacity filterCapacity{};
    filterCapacity.calls = 2048;
    filterCapacity.examinedRows = 16777216;
    filterCapacity.retainedRows = 1048576;
    filterCapacity.maximumExaminedRowsPerCall = 32768;
    gl::gpu::CudaPhase2FilterSortBuffer filter(filterCapacity);
    ASSERT_GE(filter.fixedAllocationBytes(), 16834560ull);
    ASSERT_TRUE(filter.fixedAllocationBytes() < 64ull * 1024ull * 1024ull);

    gl::gpu::Phase2GrowthCapacity capacity{};
    capacity.calls = 2048;
    capacity.retainedRows = 1048576;
    capacity.frontierRecords = 1048576;
    capacity.acceptedEvents = 2097152;
    capacity.rawRequests = 131072;
    capacity.prefixPayloadValues = 17825792;
    capacity.prefixVariableValues = 5505024;
    capacity.prefixSecondaryValues = 1310720;
    capacity.candidateWindowRecords = 16777216;
    gl::gpu::CudaPhase2GrowthBuffer buffer(capacity);
    ASSERT_TRUE(buffer.fixedAllocationBytes() > 800097352ull);
    ASSERT_TRUE(buffer.fixedAllocationBytes() < 1024ull * 1024ull * 1024ull);
    ASSERT_EQ(buffer.fixedAllocationBytes(), 812965447ull);

    gl::gpu::Phase2OrderingCapacity orderingCapacity{};
    orderingCapacity.events = 2097152;
    orderingCapacity.deduplicationSlots = 262144;
    orderingCapacity.requests = 131072;
    gl::gpu::CudaPhase2OrderingBuffer ordering(orderingCapacity);
    ASSERT_GE(ordering.fixedAllocationBytes(), 81788932ull);
    ASSERT_TRUE(ordering.fixedAllocationBytes() < 512ull * 1024ull * 1024ull);

    gl::gpu::Phase2EvaluationCapacity evaluationCapacity{};
    evaluationCapacity.logicalBlocks = 512;
    evaluationCapacity.requests = 131072;
    evaluationCapacity.reverseOwners = 2097152;
    evaluationCapacity.candidateOwners = 65536;
    evaluationCapacity.encodedHits = 65536;
    evaluationCapacity.localValues = 131072;
    evaluationCapacity.firingRecords = 65536;
    evaluationCapacity.generatedBytes = 2097152;
    evaluationCapacity.levelValues = 262144;
    evaluationCapacity.originDependencies = 262144;
    evaluationCapacity.markerKeys = 2048;
    evaluationCapacity.markerRemainingArgs = 1024;
    evaluationCapacity.markerArgs = 2048;
    gl::gpu::CudaPhase2EvaluationBuffer evaluation(evaluationCapacity);
    const uint64_t evaluationBytes = evaluation.fixedAllocationBytes();
    if (evaluationBytes != 37269035ull) {
        std::fprintf(stderr, "[phase2-gpu] evaluator bytes=%llu\n",
            static_cast<unsigned long long>(evaluationBytes));
    }
    ASSERT_EQ(evaluationBytes, 37269035ull);
    ASSERT_TRUE(evaluation.fixedAllocationBytes()
        < 64ull * 1024ull * 1024ull);
}
#endif
