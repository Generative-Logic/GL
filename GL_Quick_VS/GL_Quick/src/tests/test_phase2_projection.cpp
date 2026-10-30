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

#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "request_frontier_fixture.hpp"
#include "prover_fixture.hpp"
#include "hash_rule_installer_fixture.hpp"

#include "hash_request_generator_fixture.hpp"
#include "premise_pattern_fixture.hpp"
#include "compiled_definition_fixture.hpp"

#include "../gpu/phase2_projection.hpp"
#ifdef GL_CUDA
#include "../gpu/phase2_cuda.hpp"
#include "../gpu/phase2_sealing.hpp"
#include "../gpu/phase2_host_plan.hpp"
#include "../gpu/phase2_early_exit.hpp"
#endif
#include "../memory.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
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
    capacity.byteMapViews = logicalBlocks * 14;
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

/// @brief The retained sealing twin of a complete host projection, as the
///        streamed upload builds it: one window at offset zero appended by
///        `appendSealingProjection` (the only twin the production sealing
///        reads).
std::unique_ptr<gl::gpu::Phase2ProjectionArena> sealingTwin(
    const gl::gpu::Phase2ProjectionArena& complete) {
    auto twin = std::make_unique<gl::gpu::Phase2ProjectionArena>(
        complete.capacity, gl::gpu::Phase2ProjectionArena::SealingOnly{});
    twin->appendSealingProjection(complete, gl::gpu::Phase2ProjectionArena::ShardBases{});
    return twin;
}

}  // namespace

TEST(phase2_projection, growth_order_supports_configured_positions_above_8192) {
    gl::gpu::DeviceAcceptedGrowthEvent lower{};
    lower.count = 2;
    lower.filteredPositionCodes[0] = 8193;
    lower.filteredPositionCodes[1] = 1;
    auto higher = lower;
    higher.filteredPositionCodes[0] = 16384;
    ASSERT_LT(gl::gpu::growthEventPathComponent(higher, 0),
        gl::gpu::growthEventPathComponent(lower, 0));
    ASSERT_GE(gl::gpu::growthEventPathComponent(higher, 0),
        gl::ExecutionParameters::MAX_FILTERED_POSITION_CODE);
    higher.count = 1;
    higher.filteredPositionCodes[0] = UINT16_MAX;
    ASSERT_EQ(gl::gpu::growthEventPathComponent(higher, 0), UINT16_MAX - 1u);
    higher.filteredPositionCodes[0] = 0;
    ASSERT_EQ(gl::gpu::growthEventPathComponent(higher, 0), 0u);
    higher.count = 0;
    ASSERT_EQ(gl::gpu::growthEventPathComponent(higher, 0), 0u);
}

TEST(phase2_projection, growth_order_path_keys_fit_exact_radix_width) {
    const uint32_t mask = (1u << gl::gpu::kDeviceGrowthPathRadixBits) - 1u;
    uint32_t largest = 0;
    gl::gpu::DeviceAcceptedGrowthEvent event{};
    for (int32_t count = 0; count <= gl::ExecutionParameters::MAX_EXPRESSIONS; ++count) {
        event.count = count;
        for (uint32_t component = 0;
             component < gl::ExecutionParameters::MAX_EXPRESSIONS; ++component) {
            for (uint32_t code = 0; code <= UINT16_MAX; ++code) {
                event.filteredPositionCodes[component] = static_cast<uint16_t>(code);
                const uint32_t key = gl::gpu::growthEventPathComponent(event, component);
                ASSERT_EQ(key & mask, key);
                largest = std::max(largest, key);
            }
        }
    }
    ASSERT_EQ(largest, 2u * gl::ExecutionParameters::MAX_FILTERED_POSITION_CODE);
    ASSERT_GE(largest, 1u << (gl::gpu::kDeviceGrowthPathRadixBits - 1));
}

TEST(phase2_projection, candidate_node_windows_preserve_exact_coverage) {
    const uint64_t starts[] = {0, 2, 2, 5, 9, 9};
    const uint64_t ends[] = {2, 2, 5, 9, 9, 13};
    const uint32_t firstKind[] = {0, 2, 4, 5};
    const uint32_t secondKind[] = {1, 3};
    const auto empty = gl::gpu::growthCandidateNodeWindow(nullptr, 0, nullptr, nullptr, 0, 1);
    ASSERT_EQ(empty.begin, 0u);
    ASSERT_EQ(empty.count, 0u);
    // Exhaustive integer windows exercise gaps, boundary touches and empty spans.
    for (uint64_t begin = 0; begin < 15; ++begin) {
        for (uint64_t end = begin + 1; end <= 16; ++end) {
            for (bool first : {false, true}) {
                const uint32_t* indices = first ? firstKind : secondKind;
                const uint32_t count = first ? 4u : 2u;
                const auto window = gl::gpu::growthCandidateNodeWindow(
                    indices, count, starts, ends, begin, end);
                ASSERT_TRUE(window.begin <= count);
                ASSERT_TRUE(window.count <= count - window.begin);
                for (uint32_t index = 0; index < count; ++index) {
                    const uint32_t node = indices[index];
                    if (starts[node] == ends[node]) continue;
                    const bool overlaps = starts[node] < end && ends[node] > begin;
                    const bool selected = index >= window.begin && index - window.begin < window.count;
                    ASSERT_EQ(selected, overlaps);
                }
            }
        }
    }
    const uint64_t large = 1ull << 40;
    const uint64_t largeStarts[] = {large, large + 2, large + 9};
    const uint64_t largeEnds[] = {large + 2, large + 9, large + 15};
    const uint32_t largeIndices[] = {0, 1, 2};
    const auto middle = gl::gpu::growthCandidateNodeWindow(
        largeIndices, 3, largeStarts, largeEnds, large + 2, large + 9);
    ASSERT_EQ(middle.begin, 1u);
    ASSERT_EQ(middle.count, 1u);
}

TEST(phase2_projection, secondary_argument_mask_uses_first_iterative_identity) {
    gl::IntEncodedExpr expression{};
    ASSERT_EQ(gl::gpu::growthDistinctSecondaryArgumentMask(expression), 0u);
    expression.arity = gl::ExecutionParameters::MAX_ARITY;
    for (int32_t argument = 0; argument < expression.arity; ++argument) {
        expression.argFullId[argument] = 100 + argument;
        expression.argIteration[argument] = 0;
    }
    ASSERT_EQ(gl::gpu::growthDistinctSecondaryArgumentMask(expression), 0xffffu);
    expression.argIteration[0] = -1;
    expression.argFullId[1] = expression.argFullId[0];
    expression.argFullId[15] = expression.argFullId[1];
    expression.argIteration[2] = -2;
    expression.argFullId[3] = expression.argFullId[2];
    expression.argIteration[3] = 5;
    ASSERT_EQ(gl::gpu::growthDistinctSecondaryArgumentMask(expression), 0x7ffau);
    for (int32_t argument = 0; argument < expression.arity; ++argument) {
        expression.argFullId[argument] = 42;
        expression.argIteration[argument] = -1;
    }
    ASSERT_EQ(gl::gpu::growthDistinctSecondaryArgumentMask(expression), 0u);
    expression.argIteration[7] = 0;
    expression.argIteration[15] = 4;
    ASSERT_EQ(gl::gpu::growthDistinctSecondaryArgumentMask(expression), 0x80u);
}

// A continuation word keeps the polarity mask and up to two bound argument
// slots; unpacking restores them exactly, and an unused gate reads slot -1.
TEST(phase2_projection, next_predicate_word_round_trips_polarity_and_gates) {
    const uint32_t bare = gl::gpu::packNextPredicateWord(2, -1, 0, -1, 0);
    ASSERT_EQ(bare, 2u);
    const auto bareGate = gl::gpu::unpackNextPredicateWord(bare);
    ASSERT_EQ(bareGate.polarityMask, 2u);
    ASSERT_EQ(bareGate.slot[0], -1);
    ASSERT_EQ(bareGate.slot[1], -1);

    const uint32_t one = gl::gpu::packNextPredicateWord(3, 0, 1, -1, 0);
    const auto oneGate = gl::gpu::unpackNextPredicateWord(one);
    ASSERT_EQ(oneGate.polarityMask, 3u);
    ASSERT_EQ(oneGate.slot[0], 0);
    ASSERT_EQ(oneGate.variable[0], 1);
    ASSERT_EQ(oneGate.slot[1], -1);

    const int32_t lastSlot = gl::ExecutionParameters::MAX_ARITY - 1;
    const gl::NameId lastVariable = gl::ExecutionParameters::MAX_KEY_SLOTS;
    const uint32_t two = gl::gpu::packNextPredicateWord(1, 4, 7, lastSlot, lastVariable);
    const auto twoGate = gl::gpu::unpackNextPredicateWord(two);
    ASSERT_EQ(twoGate.polarityMask, 1u);
    ASSERT_EQ(twoGate.slot[0], 4);
    ASSERT_EQ(twoGate.variable[0], 7);
    ASSERT_EQ(twoGate.slot[1], lastSlot);
    ASSERT_EQ(twoGate.variable[1], lastVariable);
}

// Bindings: the first key initializes each slot (prefix variable or -1), a
// disagreeing or fresh argument of a later key clears the slot for good, slots
// at or above the arity stay -1.
TEST(phase2_projection, next_premise_bindings_keep_only_agreeing_prefix_variables) {
    int16_t bound[gl::ExecutionParameters::MAX_ARITY];
    const gl::NameId first[] = {2, 5, 1};          // prefix has 3 variables
    gl::gpu::mergeNextPremiseBindings(bound, 3, first, 3, true);
    ASSERT_EQ(bound[0], 2);
    ASSERT_EQ(bound[1], -1);                       // fresh variable 5 > 3
    ASSERT_EQ(bound[2], 1);
    for (int32_t slot = 3; slot < gl::ExecutionParameters::MAX_ARITY; ++slot)
        ASSERT_EQ(bound[slot], -1);

    const gl::NameId agreeing[] = {2, 1, 1};
    gl::gpu::mergeNextPremiseBindings(bound, 3, agreeing, 3, false);
    ASSERT_EQ(bound[0], 2);
    ASSERT_EQ(bound[1], -1);                       // cleared slots never return
    ASSERT_EQ(bound[2], 1);

    const gl::NameId disagreeing[] = {3, 1, 4};
    gl::gpu::mergeNextPremiseBindings(bound, 3, disagreeing, 3, false);
    ASSERT_EQ(bound[0], -1);                       // other prefix variable
    ASSERT_EQ(bound[2], -1);                       // fresh variable
}

// The device gate resolves bound variables against the node's distinct
// normalization variables and admits a statement only with the licensed sign
// and the required argument identifiers; an unbound word checks the sign only.
TEST(phase2_projection, growth_argument_gate_requires_bound_values) {
    const gl::NameId variables[] = {900, 41, 42, 43};   // pool; the prefix uses 41..43
    gl::gpu::DeviceGrowthPrefix prefix{};
    prefix.variableOffset = 1;
    prefix.variableCount = 3;
    const uint32_t word = gl::gpu::packNextPredicateWord(1, 1, 3, -1, 0);
    const gl::gpu::GrowthArgumentGate gate = gl::gpu::growthArgumentGate(word, prefix, variables);
    ASSERT_EQ(gate.slot[0], 1);
    ASSERT_EQ(gate.required[0], 43);

    gl::IntEncodedExpr expression{};
    expression.arity = 2;
    expression.negation = 0;
    expression.argId[0] = 7;
    expression.argId[1] = 43;
    ASSERT_TRUE(gate.admits(expression));
    expression.argId[1] = 42;
    ASSERT_FALSE(gate.admits(expression));        // other prefix variable's value
    expression.argId[1] = 43;
    expression.negation = 1;
    ASSERT_FALSE(gate.admits(expression));        // unlicensed sign

    const gl::gpu::GrowthArgumentGate open = gl::gpu::growthArgumentGate(
        gl::gpu::packNextPredicateWord(2, -1, 0, -1, 0), prefix, variables);
    ASSERT_TRUE(open.admits(expression));
    expression.negation = 0;
    ASSERT_FALSE(open.admits(expression));
}

TEST(phase2_projection, predicate_ranges_preserve_original_positions) {
    const uint32_t ranks[] = {1, 1, 3, 3, 3, 7};
    uint64_t keys[8]{};
    for (uint32_t i = 0; i < 6; ++i)
        keys[i + 1] = (static_cast<uint64_t>(ranks[i])
            << gl::gpu::kDeviceFilterStatementIndexBits) | (20u + i);
    for (uint32_t start = 0; start <= 6; ++start) {
        for (uint32_t rank = 0; rank <= 8; ++rank) {
            const auto found = gl::gpu::growthPredicateRange(keys, 1, 6, start, rank);
            uint32_t expected = 0;
            for (uint32_t position = start; position < 6; ++position) {
                const bool selected = position >= found.begin && position - found.begin < found.count;
                ASSERT_EQ(selected, ranks[position] == rank);
                if (ranks[position] == rank) ++expected;
            }
            ASSERT_EQ(found.count, expected);
        }
    }
    ASSERT_EQ(gl::gpu::growthPredicateRange(nullptr, 0, 0, 0, 5).count, 0u);
}

TEST(phase2_projection, prefix_index_preserves_polarity_arguments_and_memory_kind) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    const gl::NameId main = ((body).nameMap).encode("main");
    const gl::NameId a = ((body).nameMap).encode("prefixA");
    const gl::NameId b = ((body).nameMap).encode("prefixB");
    const gl::NameId c = ((body).nameMap).encode("prefixC");
    const int32_t source = ((body).ruleInterner).encode("prefixSource");
    const auto owner = gl::packRuleOwner(source, main);
    auto& ownerScratch = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    const auto subkey = [](gl::HashMemory& memory, gl::NameId count,
                           std::initializer_list<gl::NameId> values) {
        ((memory).normalizedEncodedSubkeys).assignRun(
            gl::NormKey{count, std::vector<gl::NameId>(values)}, std::vector<gl::OwnerSet>{gl::OwnerSet{}});
    };
    const auto whole = [&](gl::HashMemory& memory, gl::NameId count,
                           std::initializer_list<gl::NameId> values) {
        const std::vector<gl::NameId> key(values);
        gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(((memory).normalizedEncodedKeys),
            count, key.data(), static_cast<int32_t>(key.size()), owner, ownerScratch);
    };
    subkey(((body).overallHashMemory), 1, {a, 0, 1, 0});
    subkey(((body).overallHashMemory), 1, {b, 0, 1, 0});
    subkey(((body).overallHashMemory), 1, {c, 0});
    subkey(((body).overallHashMemory), 2, {a, 0, 1, 0, b, 0, 1, 0});
    subkey(((body).overallHashMemory), 2, {a, 0, 1, 0, b, 1, 2, 0});
    whole(((body).overallHashMemory), 2, {a, 0, 1, 0, b, 0, 2, 0});
    whole(((body).overallHashMemory), 3, {a, 0, 1, 0, b, 0, 1, 0, c, 0});
    whole(((body).overallHashMemory), 3, {a, 0, 1, 0, b, 0, 2, 0, c, 1});
    subkey(((body).localHashMemory), 1, {a, 0, 1, 0});
    subkey(((body).localHashMemory), 1, {c, 0});
    whole(((body).localHashMemory), 2, {a, 0, 1, 0, c, 0});
    auto capacity = projectionCapacity(1, 1,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), 256);
    gl::gpu::Phase2ProjectionArena arena(capacity);
    const auto block = arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
    const auto bounds = gl::gpu::measurePhase2ProjectionUsage(body, analyzer.anchorInfo, 0x0fu, true);
    ASSERT_EQ(block.byteMapViewCount, 14u);
    ASSERT_EQ(bounds.byteMapViews, 14u);
    ASSERT_EQ(arena.byteMapEntries.size(), bounds.byteMapEntries);
    ASSERT_EQ(arena.byteKeyBytes.size(), bounds.byteKeyBytes);
    ASSERT_EQ(arena.blobBytes.size(), bounds.blobBytes);
    const auto next = [&](uint32_t memory, gl::NameId count,
                          std::initializer_list<gl::NameId> prefix) {
        std::vector<gl::NameId> query{count, static_cast<gl::NameId>(prefix.size())};
        query.insert(query.end(), prefix.begin(), prefix.end());
        std::vector<std::pair<gl::NameId, uint32_t>> result;
        const auto& view = arena.byteMapViews[block.byteMapViewOffset + 10 + memory];
        for (uint32_t i = 0; i < view.entryCount; ++i) {
            const auto& entry = arena.byteMapEntries[view.entryOffset + i];
            if (entry.keyLength != query.size() * sizeof(gl::NameId)
                || std::memcmp(arena.byteKeyBytes.data() + entry.keyOffset,
                    query.data(), entry.keyLength) != 0) continue;
            ASSERT_EQ(entry.blobRecordCount, 1u);
            const auto& blob = arena.blobRecords[entry.blobRecordOffset];
            for (uint32_t at = 0; at < blob.byteLength; at += 8) {
                gl::NameId core = 0;
                uint32_t mask = 0;
                std::memcpy(&core, arena.blobBytes.data() + blob.byteOffset + at, 4);
                std::memcpy(&mask, arena.blobBytes.data() + blob.byteOffset + at + 4, 4);
                result.emplace_back(core, mask);
            }
        }
        return result;
    };
    const std::vector<std::pair<gl::NameId, uint32_t>> bothB{{b, 3}};
    const std::vector<std::pair<gl::NameId, uint32_t>> positiveC{{c, 1}};
    const std::vector<std::pair<gl::NameId, uint32_t>> negativeC{{c, 2}};
    ASSERT_TRUE(next(0, 1, {a, 0, 1, 0}) == bothB);
    ASSERT_TRUE(next(0, 2, {a, 0, 1, 0, b, 0, 1, 0}) == positiveC);
    ASSERT_TRUE(next(0, 2, {a, 0, 1, 0, b, 0, 2, 0}) == negativeC);
    ASSERT_TRUE(next(1, 1, {a, 0, 1, 0}) == positiveC);
    ASSERT_TRUE(next(2, 0, {}).empty());
    ASSERT_TRUE(next(0, 1, {b, 0, 1, 0}).empty());
    ASSERT_EQ(next(0, 0, {}).size(), 3u);
}

TEST(phase2_projection, shared_key_sizing_and_reverse_alias_preserve_membership) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    ((body).nameMap).encode("main");
    const gl::NameId core = ((body).nameMap).encode("sharedProjectionKey");
    const gl::NormKey key{1, {core, 0}};
    gl::OwnerSet first;
    gl::OwnerSet second;
    second.hasLooseOwner = true;
    ((((body).overallHashMemory)).normalizedEncodedSubkeys).assignRun(key, {first});
    ((((body).localHashMemory)).normalizedEncodedSubkeys).assignRun(key, {second});
    ((((body).workingMemory)).normalizedEncodedSubkeys).assignRun(key, {first});
    const gl::Int16SetKey remaining{{core}};
    ((((body).overallHashMemory)).remainingArgsNormalizedEncodedMap).assignRun(remaining, {key});
    const auto shared = gl::gpu::measurePhase2SharedKeyUsage(body, 0x0bu, true);
    const auto bounds = gl::gpu::measurePhase2ProjectionUsage(body, analyzer.anchorInfo, 0x0bu, true);
    const uint64_t expectedBytes = gl::Codec<gl::NormKey>::encode(key).size()
        + gl::Codec<gl::Int16SetKey>::encode(remaining).size() + 2 * sizeof(gl::NameId);
    ASSERT_EQ(shared.byteKeyBytes, expectedBytes);
    ASSERT_EQ(shared.byteMapEntries, 3u);
    ASSERT_EQ(shared.byteMapSlots, 6u);
    ASSERT_EQ(shared.blobRecords, 3u);
    ASSERT_EQ(shared.blobBytes, 24u);
    auto capacity = projectionCapacity(2, 1,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1) * 2, 512);
    capacity.byteKeyBytes = static_cast<uint32_t>(bounds.byteKeyBytes * 2);
    gl::gpu::Phase2ProjectionArena arena(capacity);
    const auto block = arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0bu, true);
    ASSERT_EQ(arena.byteKeyBytes.size(), expectedBytes);
    ASSERT_EQ(arena.blobBytes.size(), bounds.blobBytes);
    ASSERT_EQ(arena.reverseKeyBytes.size(), 0u);
    const auto& overall = arena.byteMapEntries[arena.byteMapViews[block.byteMapViewOffset + 1].entryOffset];
    const auto& local = arena.byteMapEntries[arena.byteMapViews[block.byteMapViewOffset + 3].entryOffset];
    const auto& working = arena.byteMapEntries[arena.byteMapViews[block.byteMapViewOffset + 7].entryOffset];
    ASSERT_EQ(overall.keyOffset, local.keyOffset);
    ASSERT_EQ(overall.keyOffset, working.keyOffset);
    ASSERT_TRUE(overall.blobRecordOffset != local.blobRecordOffset);
    const auto& firstBlob = arena.blobRecords[overall.blobRecordOffset];
    const auto& secondBlob = arena.blobRecords[local.blobRecordOffset];
    ASSERT_TRUE(arena.blobBytes[firstBlob.byteOffset] != arena.blobBytes[secondBlob.byteOffset]);
    ASSERT_EQ(arena.byteMapViews[block.byteMapViewOffset + 5].entryCount, 0u);
    const auto& forward = arena.byteMapEntries[arena.byteMapViews[block.byteMapViewOffset + 9].entryOffset];
    const auto& reverse = arena.reverseMapEntries[arena.reverseMapViews[block.reverseMapViewOffset].entryOffset];
    ASSERT_EQ(reverse.keyOffset, arena.blobRecords[forward.blobRecordOffset].byteOffset);
    const auto secondBlock = arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0bu, true);
    ASSERT_EQ(arena.byteKeyBytes.size(), expectedBytes * 2);
    const auto& nextKey = arena.byteMapEntries[arena.byteMapViews[secondBlock.byteMapViewOffset + 1].entryOffset];
    ASSERT_TRUE(nextKey.keyOffset != overall.keyOffset);
    ASSERT_EQ(((((body).overallHashMemory)).normalizedEncodedSubkeys).count(), 1);
    ASSERT_EQ(((((body).localHashMemory)).normalizedEncodedSubkeys).count(), 1);
}

TEST(phase2_projection, shared_key_empty_and_assertion_probe) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    const auto empty = gl::gpu::measurePhase2SharedKeyUsage(body, 1u, true);
    ASSERT_EQ(empty.byteKeyBytes, 0u);
    ASSERT_EQ(empty.byteMapEntries, 0u);
    ASSERT_EQ(empty.blobBytes, 0u);
    std::string mode;
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    ASSERT_EQ(_dupenv_s(&value, &length, "GL_TEST_INVALID_PROJECTION_KEYS"), 0);
    if (value != nullptr) { mode = value; std::free(value); }
#else
    if (const char* value = std::getenv("GL_TEST_INVALID_PROJECTION_KEYS")) mode = value;
#endif
    if (mode == "mask") {
        gl::gpu::measurePhase2SharedKeyUsage(body, 0u, true);
        ASSERT_TRUE(false);
    }
    else if (mode == "missing_arity") {
        ((body).nameMap).encode("main");
        const gl::NameId core = ((body).nameMap).encode("missingSingleton");
        const gl::NormKey key{2, {core, 0, core, 0}};
        ((((body).overallHashMemory)).normalizedEncodedSubkeys).assignRun(key, {gl::OwnerSet{}});
        gl::gpu::measurePhase2SharedKeyUsage(body, 1u, true);
        ASSERT_TRUE(false);
    }
    else {
        ASSERT_TRUE(mode.empty());
    }
}

TEST(phase2_projection, signed_core_prefilter_preserves_order_and_rebuilds) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    auto first = gl::encodeExpression(gl::StrSpan("(p[a])", 6),
        gl::StrSpan("main", 4), ((body).nameMap));
    auto negative = gl::encodeExpression(gl::StrSpan("!(p[b])", 7),
        gl::StrSpan("main", 4), ((body).nameMap));
    auto last = gl::encodeExpression(gl::StrSpan("(p[b])", 6),
        gl::StrSpan("main", 4), ((body).nameMap));
    ((body).intEncodedStatements).push_back(first);
    ((body).intEncodedStatements).push_back(negative);
    ((body).intEncodedStatements).push_back(last);
    gl::tests::admitEveryStatementShape(body);
    const gl::NameId key[] = { first.nameId, 0, 1, 0 };
    gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
        ((((body).localHashMemory)).normalizedEncodedSubkeys), 1, key, 4);
    std::set<std::pair<gl::NameId, int32_t>> cores;
    gl::forEachPremiseCore(body, [&](gl::NameId core, int32_t polarity) {
        cores.emplace(core, polarity);
    });
    ASSERT_EQ(cores.size(), 1u);
    ASSERT_TRUE(cores.count({first.nameId, 0}) == 1);
    ASSERT_TRUE(cores.count({first.nameId, 1}) == 0);
    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(1, 2,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), 1024));
    const auto projected = arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(projected.statementCount, 2u);
    ASSERT_EQ(arena.statements[0].originalId, first.originalId);
    ASSERT_EQ(arena.statements[1].originalId, last.originalId);
    ASSERT_EQ(arena.projectedStatementIndex(0, 0), 0);
    ASSERT_EQ(arena.projectedStatementIndex(0, 2), 1);
    // Source ordinals survive the in-place adoption of a slice shard.
    const gl::gpu::Phase2ProjectionCapacity oneBlock = projectionCapacity(1, 2,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), 1024);
    gl::gpu::Phase2ProjectionArena merged(oneBlock, oneBlock);
    gl::gpu::Phase2ProjectionArena slice(
        oneBlock, merged, gl::gpu::Phase2ProjectionCapacity{});
    slice.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    gl::gpu::Phase2ProjectionArena* shards[] = { &slice };
    uint32_t firstBlock[1]{ 5 };
    gl::gpu::Phase2ProjectionWorkers mergeWorkers(2);
    merged.adoptShards(shards, 1, firstBlock,
        gl::gpu::Phase2ProjectionArena::ShardBases{}, mergeWorkers);
    ASSERT_EQ(firstBlock[0], 0u);
    ASSERT_EQ(merged.projectedStatementIndex(0, 2), 1);
    arena.clear();
    ((((body).localHashMemory)).normalizedEncodedSubkeys).resetToFresh();
    const auto empty = arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(empty.statementCount, 0u);
    arena.clear();
    const gl::NameId negKey[] = { first.nameId, 1, 1, 0 };
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).workingMemory)).normalizedEncodedKeys), 1, negKey, 4, 1,
        gl::genScratchArenas().forSlot(0));
    const auto refreshed = arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(refreshed.statementCount, 1u);
    ASSERT_EQ(arena.statements[0].originalId, negative.originalId);
}

TEST(phase2_projection, upload_is_used_as_premise_only) {
    // The device receives exactly the rows whose SIGNED core is used as a
    // premise: `(p, positive)` is the only premise core here, so the negated
    // `p` row and both `q` rows stay on the host.
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    auto pPos = gl::encodeExpression(gl::StrSpan("(p[a])", 6),
        gl::StrSpan("main", 4), ((body).nameMap));
    auto pNeg = gl::encodeExpression(gl::StrSpan("!(p[b])", 7),
        gl::StrSpan("main", 4), ((body).nameMap));
    auto qPos = gl::encodeExpression(gl::StrSpan("(q[c])", 6),
        gl::StrSpan("main", 4), ((body).nameMap));
    auto qNeg = gl::encodeExpression(gl::StrSpan("!(q[d])", 7),
        gl::StrSpan("main", 4), ((body).nameMap));
    ((body).intEncodedStatements).push_back(pPos);
    ((body).intEncodedStatements).push_back(pNeg);
    ((body).intEncodedStatements).push_back(qPos);
    ((body).intEncodedStatements).push_back(qNeg);
    gl::tests::admitEveryStatementShape(body);
    const gl::NameId key[] = { pPos.nameId, 0, 1, 0 };
    gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
        ((((body).localHashMemory)).normalizedEncodedSubkeys), 1, key, 4);
    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(1, 4,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), 1024));
    const auto projected = arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(projected.statementCount, 1u);
    ASSERT_EQ(arena.statements[0].originalId, pPos.originalId);
    ASSERT_EQ(arena.projectedStatementIndex(0, 0), 0);
}

TEST(phase2_projection, upload_requires_a_live_premise_pattern) {
    // A row whose signed core is a premise core is uploaded only when a live
    // rule premise accepts it on its u_ literals and scope: the premise fixes
    // `k`, so `(p[k])` is uploaded and `(p[j])` stays on the host.
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    const auto pj = gl::encodeExpression(gl::StrSpan("(p[j])", 6),
        gl::StrSpan("main", 4), ((body).nameMap));
    const auto pk = gl::encodeExpression(gl::StrSpan("(p[k])", 6),
        gl::StrSpan("main", 4), ((body).nameMap));
    const auto premise = gl::encodeExpression(gl::StrSpan("(p[u_k])", 8),
        gl::StrSpan("main", 4), ((body).nameMap));
    ((body).intEncodedStatements).push_back(pj);
    ((body).intEncodedStatements).push_back(pk);
    const gl::NameId key[] = { pk.nameId, 0, 1, 0 };
    gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
        ((((body).localHashMemory)).normalizedEncodedSubkeys), 1, key, 4);
    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(1, 2,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), 1024));
    ASSERT_EQ(arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo,
        analyzer.parameters).statementCount, 0u);
    arena.clear();
    gl::tests::registerPremisePattern(((body).localHashMemory), premise,
        gl::NameMap::MAIN_ID, 1);
    const auto projected = arena.appendLogicalBlock(body, analyzer.definitions,
        analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(projected.statementCount, 1u);
    ASSERT_EQ(arena.statements[0].originalId, pk.originalId);
    ASSERT_EQ(arena.projectedStatementIndex(0, 1), 0);
}

TEST(phase2_projection, phase1_scope_cleanup_precedes_projection) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    ((body).isActive) = true;
    ((body).level) = 0;
    const auto closed = ((body).nameMap).encodePush(gl::NameMap::MAIN_ID, "closed_case");
    const auto child = ((body).nameMap).encodePush(closed, "nested_case");
    const auto sibling = ((body).nameMap).encodePush(gl::NameMap::MAIN_ID, "live_case");
    const int levels[] = {0};
    for (auto validity : {gl::NameMap::MAIN_ID, closed, child, sibling}) {
        const std::string expression = "(p[" + std::to_string(validity) + "])";
        const auto row = gl::encodeExpression(gl::EncodedExpression(expression,
            ((body).nameMap).decode(validity)), ((body).nameMap));
        ((body).intEncodedStatements).push_back(row);
        ((body).intLocalEncodedStatements).push_back(row);
        ((body).intLocalEncodedStatementsDelta).push_back(row);
        const auto key = gl::packStatementKey(row.originalId, validity);
        ((body).intLocalEncodedStatementsSet).mint(key);
        gl::upsertStatementKey(((body).intKnownStatements), key, true);
        ((body).intStatementLevelsMap).assignSet(key, levels, 1);
        const std::string scope = ((body).nameMap).decode(validity);
        gl::addOriginEncoded(((body).exprOriginMap), ((body).originInterner),
            gl::StrSpan(expression), gl::StrSpan(scope),
            gl::OriginTag::taskFormulation, nullptr, 0, 8);
    }
    gl::tests::admitEveryStatementShape(body);
    const auto core = ((body).nameMap).lookup("p");
    const gl::NameId key[] = {core, 0, 1, 0};
    gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
        ((((body).localHashMemory)).normalizedEncodedSubkeys), 1, key, 4);
    gl::LocalMemoryValue owner{};
    owner.validityId = closed;
    const gl::NormKey normalized{1, {core, 0, 1, 0}};
    for (auto* hash : {&((body).overallHashMemory), &((body).localHashMemory), &((body).localHashMemoryDelta)})
        ((*(hash)).encodedMap).assignRun(normalized, {owner});
    ((body).pendingWipeScopes).mint(child);
    ((body).pendingWipeScopes).mint(closed);
    analyzer.logicBlockOperations.standardProcessing(body, nullptr,
        ((body).nextIterationInternalMail), ((body).sameIterationInternalMail), -1);
    ASSERT_EQ(((body).intEncodedStatements).size(), 2);
    ASSERT_TRUE(((((body).overallHashMemory)).encodedMap).empty());
    ASSERT_TRUE(((((body).localHashMemory)).encodedMap).empty());
    ASSERT_TRUE(((((body).localHashMemoryDelta)).encodedMap).empty());
    ASSERT_TRUE(((body).intValidityNamesToFilter).contains(closed));
    ASSERT_TRUE(((body).intValidityNamesToFilter).contains(child));
    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(1, 4,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), 2048));
    const auto projected = arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(projected.statementCount, 2u);
    for (uint32_t i = 0; i < projected.statementCount; ++i) {
        ASSERT_FALSE(((body).nameMap).ancContains(arena.statements[i].validityId, closed));
    }
}

TEST(phase2_projection, backend_defaults_to_processor_and_accepts_explicit_processor) {
    // The product default is the processor route (`use_gpu` false, no
    // shipped config pins a batch); it keeps its SSD permission and pages.
    gl::tests::ProverFixture defaultAnalyzer("FTA");
    ASSERT_FALSE(defaultAnalyzer.parameters.use_gpu);
    ASSERT_TRUE(defaultAnalyzer.parameters.allow_ssd_deload);
    ASSERT_EQ(defaultAnalyzer.phase2Backend, gl::Phase2Backend::cpu);
    // The explicit processor override is accepted as well.
    gl::tests::ProverFixture cpuAnalyzer("FTA", gl::Phase2Backend::cpu);
    ASSERT_EQ(cpuAnalyzer.phase2Backend, gl::Phase2Backend::cpu);
    ASSERT_TRUE(cpuAnalyzer.parameters.allow_ssd_deload);
}

#ifdef GL_CUDA
// A CUDA-route test: constructing a CUDA batch starts the CUDA context
// warm-up, which needs a device, so the processor-route gate of a machine
// without a GPU does not select it. A build without CUDA support refuses the
// selection at construction instead.
CUDA_TEST(phase2_projection, backend_accepts_explicit_cuda) {
    // `main.py --GPU` arrives as the explicit CUDA override; a CUDA batch
    // keeps the SSD permission and pages like a processor batch.
    gl::tests::ProverFixture cudaAnalyzer("FTA", gl::Phase2Backend::cuda);
    ASSERT_EQ(cudaAnalyzer.phase2Backend, gl::Phase2Backend::cuda);
    ASSERT_TRUE(cudaAnalyzer.parameters.allow_ssd_deload);
}
#endif

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
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.frontierRecords, 3670016u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.acceptedEvents, 6291456u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.prefixPayloadValues, 109051904u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.prefixVariableValues, 29360128u);
    ASSERT_EQ(gl::gpu::kFullRunPhase2GrowthCapacity.prefixSecondaryValues, 6291456u);
    // Every resized pool retains at least twenty percent over its recorded full-corpus peak.
    const auto& commonGrowth = gl::gpu::kFullRunPhase2GrowthCapacity;
    ASSERT_TRUE(5ULL * commonGrowth.acceptedEvents >= 6ULL * 5214880ULL);
    ASSERT_TRUE(5ULL * commonGrowth.frontierRecords >= 6ULL * 2888479ULL);
    ASSERT_TRUE(5ULL * commonGrowth.prefixPayloadValues >= 6ULL * 88141734ULL);
    ASSERT_TRUE(5ULL * commonGrowth.prefixVariableValues >= 6ULL * 23900106ULL);
    ASSERT_TRUE(5ULL * commonGrowth.prefixSecondaryValues >= 6ULL * 4426243ULL);
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
    ASSERT_EQ(sizeof(gl::gpu::DeviceEvaluationOwnerWork), 8u);
#endif
}

TEST(phase2_projection, rows_names_parents_and_ranks_are_lossless) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    const gl::NameId mainId = ((body).nameMap).encode("main");
    const gl::NameId zetaId = ((body).nameMap).encode("zeta");
    const gl::NameId alphaId = ((body).nameMap).encode("alpha");
    const gl::NameId childId = ((body).nameMap).encodePush(mainId, "child");
    const gl::NameId originalA = ((body).nameMap).encode("(zeta[alpha])");
    const gl::NameId originalB = ((body).nameMap).encode("(alpha[zeta])");
    ((body).nameMap).encode("prefix");
    ((body).nameMap).encode("prefix0");
    ((body).nameMap).encode("prefix00");
    ((body).nameMap).encode("prefix01");
    ((body).nameMap).encode("prefix1");

    const gl::IntEncodedExpr first =
        projectionStatement(zetaId, originalA, mainId, alphaId);
    const gl::IntEncodedExpr second =
        projectionStatement(alphaId, originalB, childId, zetaId);
    ((body).intEncodedStatements).push_back(first);
    ((body).intEncodedStatements).push_back(second);

    uint32_t byteCount = 0;
    for (gl::NameId id = 1; id <= ((body).nameMap).nameCount(); ++id)
        byteCount += static_cast<uint32_t>(((body).nameMap).decodeView(id).len);

    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(
        1,
        2,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1),
        byteCount));
    const gl::gpu::DeviceLogicalBlockProjection projected =
        arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);

    ASSERT_EQ(projected.statementOffset, 0u);
    ASSERT_EQ(projected.statementCount, 0u);
    ASSERT_EQ(projected.nameRecordOffset, 0u);
    ASSERT_EQ(projected.nameRecordCount,
              static_cast<uint32_t>(((body).nameMap).nameCount() + 1));
    ASSERT_EQ(projected.nameByteCount, byteCount);
    ASSERT_TRUE(arena.statements.empty());

    for (gl::NameId id = 1; id <= ((body).nameMap).nameCount(); ++id) {
        ASSERT_EQ(projectedName(arena, projected, id), ((body).nameMap).decode(id));
        ASSERT_EQ(arena.nameRecords[static_cast<uint32_t>(id)].parentId,
                  ((body).nameMap).parentOf(id));
    }
    ASSERT_EQ(arena.nameRecords[static_cast<uint32_t>(childId)].parentId, mainId);
    ASSERT_LT(arena.nameRecords[static_cast<uint32_t>(alphaId)].decodedLexRank,
              arena.nameRecords[static_cast<uint32_t>(zetaId)].decodedLexRank);

    std::vector<gl::NameId> expectedOrder;
    for (gl::NameId id = 1; id <= ((body).nameMap).nameCount(); ++id)
        expectedOrder.push_back(id);
    std::sort(expectedOrder.begin(), expectedOrder.end(),
        [&body](gl::NameId left, gl::NameId right) {
            return gl::compareSpans(((body).nameMap).decodeView(left),
                                    ((body).nameMap).decodeView(right)) < 0;
        });
    for (uint32_t rank = 0; rank < expectedOrder.size(); ++rank) {
        const gl::NameId id = expectedOrder[rank];
        ASSERT_EQ(arena.nameRecords[static_cast<uint32_t>(id)].decodedLexRank,
                  rank);
    }
}

TEST(phase2_projection, append_offsets_and_clear_reuse_fixed_storage) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory firstBody;
    const gl::NameId firstMain = ((firstBody).nameMap).encode("main");
    const gl::NameId firstName = ((firstBody).nameMap).encode("beta");
    ((firstBody).intEncodedStatements).push_back(
        projectionStatement(firstName, firstName, firstMain, firstName));

    gl::Memory secondBody;
    const gl::NameId secondMain = ((secondBody).nameMap).encode("main");
    const gl::NameId secondName = ((secondBody).nameMap).encode("gamma");
    ((secondBody).intEncodedStatements).push_back(
        projectionStatement(secondName, secondName, secondMain, secondName));

    uint32_t bytes = 0;
    for (gl::NameId id = 1; id <= ((firstBody).nameMap).nameCount(); ++id)
        bytes += static_cast<uint32_t>(((firstBody).nameMap).decodeView(id).len);
    for (gl::NameId id = 1; id <= ((secondBody).nameMap).nameCount(); ++id)
        bytes += static_cast<uint32_t>(((secondBody).nameMap).decodeView(id).len);

    const uint32_t records =
        static_cast<uint32_t>(((firstBody).nameMap).nameCount()
            + ((secondBody).nameMap).nameCount() + 2);
    gl::gpu::Phase2ProjectionArena arena(
        projectionCapacity(2, 2, records, bytes));
    const gl::IntEncodedExpr* statementAllocation = arena.statements.data();
    const gl::gpu::DeviceNameRecord* nameAllocation = arena.nameRecords.data();
    const char* byteAllocation = arena.nameBytes.data();

    const auto firstProjection = arena.appendLogicalBlock(firstBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    const auto secondProjection = arena.appendLogicalBlock(secondBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(secondProjection.statementOffset,
              firstProjection.statementOffset + firstProjection.statementCount);
    ASSERT_EQ(secondProjection.nameRecordOffset,
              firstProjection.nameRecordOffset + firstProjection.nameRecordCount);
    ASSERT_EQ(secondProjection.nameByteOffset,
              firstProjection.nameByteOffset + firstProjection.nameByteCount);
    ASSERT_EQ(arena.statements.size(), 0u);
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

    const auto reused = arena.appendLogicalBlock(firstBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(reused.statementOffset, 0u);
    ASSERT_EQ(reused.nameRecordOffset, 0u);
    ASSERT_EQ(reused.nameByteOffset, 0u);
    ASSERT_EQ(arena.timing.logicalBlocks, 1u);
}

TEST(phase2_projection, identical_read_only_string_columns_share_exactly) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory firstBody;
    const gl::NameId firstMain = ((firstBody).nameMap).encode("main");
    ((firstBody).nameMap).encodePush(firstMain, "child");
    ((firstBody).ruleInterner).encode("rule_a");

    gl::Memory identicalBody;
    const gl::NameId identicalMain = ((identicalBody).nameMap).encode("main");
    ((identicalBody).nameMap).encodePush(identicalMain, "child");
    ((identicalBody).ruleInterner).encode("rule_a");

    gl::Memory differentParentBody;
    ((differentParentBody).nameMap).encode("main");
    ((differentParentBody).nameMap).encode("child");
    ((differentParentBody).ruleInterner).encode("rule_a");

    gl::Memory differentRuleBody;
    const gl::NameId differentRuleMain =
        ((differentRuleBody).nameMap).encode("main");
    ((differentRuleBody).nameMap).encodePush(differentRuleMain, "child");
    ((differentRuleBody).ruleInterner).encode("rule_b");

    gl::gpu::Phase2ProjectionArena arena(
        projectionCapacity(4, 4, 16, 128));
    const auto first = arena.appendLogicalBlock(firstBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    const std::size_t firstNameRecords = arena.nameRecords.size();
    const std::size_t firstNameBytes = arena.nameBytes.size();
    const std::size_t firstNameSlots = arena.nameSlots.size();
    const std::size_t firstRuleRecords = arena.ruleStringRecords.size();
    const std::size_t firstRuleBytes = arena.ruleStringBytes.size();

    const auto identical =
        arena.appendLogicalBlock(identicalBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
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
        arena.appendLogicalBlock(differentParentBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_NE(differentParent.nameRecordOffset, first.nameRecordOffset);
    ASSERT_EQ(differentParent.ruleStringRecordOffset,
              first.ruleStringRecordOffset);

    const auto differentRule =
        arena.appendLogicalBlock(differentRuleBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(differentRule.nameRecordOffset, first.nameRecordOffset);
    ASSERT_NE(differentRule.ruleStringRecordOffset,
              first.ruleStringRecordOffset);
}

// Sharing compares a new block against the arena's projected copy of the
// earlier block, so a change to the earlier block after its projection (or its
// deload) cannot affect the decision.
TEST(phase2_projection, sharing_compares_the_projected_copy_not_the_live_block) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory firstBody;
    const gl::NameId firstMain = ((firstBody).nameMap).encode("main");
    ((firstBody).nameMap).encodePush(firstMain, "child");
    ((firstBody).ruleInterner).encode("rule_a");
    gl::Memory laterBody;
    const gl::NameId laterMain = ((laterBody).nameMap).encode("main");
    ((laterBody).nameMap).encodePush(laterMain, "child");
    ((laterBody).ruleInterner).encode("rule_a");

    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(4, 4, 16, 128));
    const auto first = arena.appendLogicalBlock(firstBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    // The first block changes after its projection.
    ((firstBody).nameMap).encode("extra");
    ((firstBody).ruleInterner).encode("rule_extra");
    const auto later = arena.appendLogicalBlock(laterBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(later.nameRecordOffset, first.nameRecordOffset);
    ASSERT_EQ(later.nameByteOffset, first.nameByteOffset);
    ASSERT_EQ(later.ruleStringRecordOffset, first.ruleStringRecordOffset);
    ASSERT_EQ(later.ruleStringByteOffset, first.ruleStringByteOffset);
}

TEST(phase2_projection, shard_merge_rebases_every_semantic_column_exactly) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["theta"] = gl::LogicalEntity(
        "atomic", {}, "theta", 1);

    gl::Memory prefixBody;
    const gl::NameId prefixMain = ((prefixBody).nameMap).encode("main");
    const gl::NameId prefixName = ((prefixBody).nameMap).encode("prefix");
    prefixBody.setExprKey("prefix_block");
    ((prefixBody).intEncodedStatements).push_back(projectionStatement(
        prefixName, prefixName, prefixMain, prefixName));

    gl::Memory richBody;
    const gl::NameId mainId = ((richBody).nameMap).encode("main");
    const gl::NameId nameId = ((richBody).nameMap).encode("theta");
    richBody.setExprKey("rich_block");
    const gl::IntEncodedExpr statement = projectionStatement(
        nameId, nameId, mainId, nameId);
    ((richBody).intEncodedStatements).push_back(statement);
    ((richBody).intLocalEncodedStatements).push_back(statement);
    ((richBody).intLocalEncodedStatementsDelta).push_back(statement);
    ((richBody).intExternalStatements).push_back(statement);

    const int32_t ruleHeadId = ((richBody).ruleInterner).encode("(theta[1])");
    const int32_t ruleSourceId = ((richBody).ruleInterner).encode(
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
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((richBody).overallHashMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((richBody).overallHashMemory)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    ((((richBody).overallHashMemory)).encodedMap).assignRun(
        normKey, std::vector<gl::LocalMemoryValue>{ value });
    ((((richBody).overallHashMemory)).remainingArgsNormalizedEncodedMap).assignRun(
        remainingKey, std::vector<gl::NormKey>{ normKey });
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((richBody).localHashMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((richBody).localHashMemory)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((richBody).localHashMemoryDelta)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((richBody).localHashMemoryDelta)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((richBody).workingMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((richBody).workingMemory)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    ((((richBody).overallHashMemory)).productsOfRecursionIds).mint(nameId);
    const gl::StatementKey statementKey{ nameId, mainId };
    const int64_t packedStatementKey = gl::packStatementKey(nameId, mainId);
    ((richBody).intKnownStatements).upsert(
        statementKey, gl::StatementFlags{ true, true });
    ((richBody).intLocalEncodedStatementsSet).mint(packedStatementKey);
    const int levels[1] = { 4 };
    ((richBody).intStatementLevelsMap).assignSet(statementKey, levels, 1);
    ((richBody).intValidityNamesToFilter).mint(mainId);
    ((richBody).frozenOrBranches).mint(nameId);
    ((richBody).intToBeProved).assignSet(statementKey, levels, 1);
    gl::tests::admitEveryStatementShape(prefixBody);
    gl::tests::admitEveryStatementShape(richBody);

    const gl::gpu::Phase2ProjectionCapacity capacity = projectionCapacity(
        2, 2, 16, 512);
    gl::gpu::Phase2ProjectionArena serial(capacity);
    serial.appendLogicalBlock(prefixBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    serial.appendLogicalBlock(richBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);

    // Two slice shards laid end to end in one backing arena write straight
    // into it; adoption closes the gap between them and rebases every column.
    const gl::gpu::Phase2ProjectionCapacity storage =
        gl::gpu::addProjectionCapacity(capacity, capacity);
    gl::gpu::Phase2ProjectionArena backing(capacity, storage);
    gl::gpu::Phase2ProjectionArena prefixShard(
        capacity, backing, gl::gpu::Phase2ProjectionCapacity{});
    gl::gpu::Phase2ProjectionArena richShard(capacity, backing, capacity);
    prefixShard.appendLogicalBlock(prefixBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    richShard.appendLogicalBlock(richBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_TRUE(!richShard.byteMapEntries.empty());
    ASSERT_TRUE(!richShard.blobRecords.empty());
    ASSERT_TRUE(!richShard.reverseMapEntries.empty());
    ASSERT_TRUE(!richShard.podMapEntries.empty());
    ASSERT_TRUE(!richShard.podRunValues.empty());
    // The shards own no column storage: their rows live in the backing arena.
    ASSERT_EQ(prefixShard.logicalBlocks.ownedBytes(), uint64_t(0));
    ASSERT_TRUE(prefixShard.hostBytes() > 0);
    ASSERT_EQ(richShard.logicalBlocks.ownedBytes(), uint64_t(0));
    ASSERT_TRUE(richShard.hostBytes() > 0);
    ASSERT_TRUE(backing.hostBytes() > 0);
    ASSERT_TRUE(richShard.statements.data()
        == backing.statements.data() + capacity.statements);
    ASSERT_TRUE(richShard.metadataBytes.data()
        == backing.metadataBytes.data() + capacity.metadataBytes);
    const uint64_t richShardTimingRows = richShard.timing.retainedStatementRows;

    gl::gpu::Phase2ProjectionArena* shards[] = { &prefixShard, &richShard };
    uint32_t firstBlocks[2]{ 7, 7 };
    gl::gpu::Phase2ProjectionWorkers adoptionWorkers(3);
    backing.adoptShards(shards, 2, firstBlocks,
        gl::gpu::Phase2ProjectionArena::ShardBases{}, adoptionWorkers);
    ASSERT_EQ(firstBlocks[0], 0u);
    ASSERT_EQ(firstBlocks[1], 1u);
    ASSERT_EQ(backing.timing.logicalBlocks, 2u);
    ASSERT_EQ(backing.logicalBlocks.size(), serial.logicalBlocks.size());
    ASSERT_EQ(backing.statements.size(), serial.statements.size());
    ASSERT_EQ(backing.nameRecords.size(), serial.nameRecords.size());
    ASSERT_EQ(backing.byteMapEntries.size(), serial.byteMapEntries.size());
    ASSERT_EQ(backing.reverseMapEntries.size(), serial.reverseMapEntries.size());
    ASSERT_EQ(backing.podMapEntries.size(), serial.podMapEntries.size());
    ASSERT_EQ(projectionChecksum(backing), projectionChecksum(serial));
    ASSERT_TRUE(backing.sourceStatementIndices == serial.sourceStatementIndices);
    ASSERT_EQ(backing.timing.retainedStatementRows, serial.timing.retainedStatementRows);
    ASSERT_TRUE(richShard.logicalBlocks.empty() && richShard.byteMapEntries.empty());
    ASSERT_TRUE(prefixShard.statements.empty() && prefixShard.metadataBytes.empty());
    ASSERT_EQ(richShard.timing.retainedStatementRows, richShardTimingRows);

    // Three shards with an empty middle one: the empty slice adopts nothing
    // and the third shard's rows close up behind the first.
    const gl::gpu::Phase2ProjectionCapacity three = projectionCapacity(3, 4, 32, 1024);
    gl::gpu::Phase2ProjectionArena serialThree(three);
    serialThree.appendLogicalBlock(prefixBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    serialThree.appendLogicalBlock(richBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    const gl::gpu::Phase2ProjectionCapacity twoSlices =
        gl::gpu::addProjectionCapacity(capacity, capacity);
    gl::gpu::Phase2ProjectionArena backingThree(
        three, gl::gpu::addProjectionCapacity(twoSlices, capacity));
    gl::gpu::Phase2ProjectionArena first(
        capacity, backingThree, gl::gpu::Phase2ProjectionCapacity{});
    gl::gpu::Phase2ProjectionArena empty(capacity, backingThree, capacity);
    gl::gpu::Phase2ProjectionArena third(capacity, backingThree, twoSlices);
    first.appendLogicalBlock(prefixBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    third.appendLogicalBlock(richBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    gl::gpu::Phase2ProjectionArena* threeShards[] = { &first, &empty, &third };
    uint32_t threeFirst[3]{ 9, 9, 9 };
    backingThree.adoptShards(threeShards, 3, threeFirst,
        gl::gpu::Phase2ProjectionArena::ShardBases{}, adoptionWorkers);
    ASSERT_EQ(threeFirst[0], 0u);
    ASSERT_EQ(threeFirst[1], 1u);
    ASSERT_EQ(threeFirst[2], 1u);
    ASSERT_EQ(backingThree.timing.logicalBlocks, 2u);
    ASSERT_EQ(projectionChecksum(backingThree), projectionChecksum(serialThree));
    ASSERT_TRUE(backingThree.sourceStatementIndices == serialThree.sourceStatementIndices);

    // Adoption at stream bases equals whole adoption followed by the separate
    // stream rebase: every reference lands at its complete-image offset in one
    // pass (D-447).
    gl::gpu::Phase2ProjectionArena::ShardBases streamBases{};
    {
        uint32_t words[24];
        for (uint32_t field = 0; field < 24; ++field) words[field] = 1000u + 37u * field;
        static_assert(sizeof(streamBases) == sizeof(words));
        std::memcpy(&streamBases, words, sizeof(words));
    }
    gl::gpu::Phase2ProjectionArena serialStreamed(capacity);
    serialStreamed.appendLogicalBlock(prefixBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    serialStreamed.appendLogicalBlock(richBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    serialStreamed.rebaseForStream(streamBases);
    gl::gpu::Phase2ProjectionArena backingStreamed(capacity, storage);
    gl::gpu::Phase2ProjectionArena prefixStreamed(
        capacity, backingStreamed, gl::gpu::Phase2ProjectionCapacity{});
    gl::gpu::Phase2ProjectionArena richStreamed(capacity, backingStreamed, capacity);
    prefixStreamed.appendLogicalBlock(prefixBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    richStreamed.appendLogicalBlock(richBody, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    gl::gpu::Phase2ProjectionArena* streamedShards[] = { &prefixStreamed, &richStreamed };
    uint32_t streamedFirst[2]{ 7, 7 };
    backingStreamed.adoptShards(streamedShards, 2, streamedFirst, streamBases, adoptionWorkers);
    ASSERT_EQ(streamedFirst[0], 0u);
    ASSERT_EQ(streamedFirst[1], 1u);
    ASSERT_EQ(projectionChecksum(backingStreamed), projectionChecksum(serialStreamed));
    ASSERT_EQ(std::memcmp(backingStreamed.logicalBlocks.data(), serialStreamed.logicalBlocks.data(),
        serialStreamed.logicalBlocks.size() * sizeof(gl::gpu::DeviceLogicalBlockProjection)), 0);
    ASSERT_EQ(backingStreamed.logicalBlocks[1].statementOffset,
        streamBases.statement + serial.logicalBlocks[1].statementOffset);
    ASSERT_TRUE(backingStreamed.byteMapSlots == serialStreamed.byteMapSlots);
    ASSERT_TRUE(backingStreamed.podMapSlots == serialStreamed.podMapSlots);
    ASSERT_TRUE(backingStreamed.reverseMapSlots == serialStreamed.reverseMapSlots);
    ASSERT_TRUE(backingStreamed.nameSlots == serialStreamed.nameSlots);
    ASSERT_TRUE(backingStreamed.sourceStatementIndices == serialStreamed.sourceStatementIndices);
}

TEST(phase2_projection, projection_workers_run_every_task_once_per_job) {
    gl::gpu::Phase2ProjectionWorkers workers(3);
    std::vector<int> hits(100, 0);
    std::atomic<uint32_t> calls{0};
    auto count = [&hits, &calls](uint32_t task) {
        ++hits[task];
        calls.fetch_add(1, std::memory_order_relaxed);
    };
    // Successive jobs of different sizes, including none and more tasks than
    // threads, reuse the same threads; every index runs exactly once per job
    // and the job's writes are visible when `run` returns.
    const uint32_t sizes[] = { 0, 1, 3, 25, 100, 7 };
    uint32_t expectedCalls = 0;
    for (const uint32_t size : sizes) {
        std::fill(hits.begin(), hits.end(), 0);
        workers.run(size, count);
        expectedCalls += size;
        for (uint32_t task = 0; task < hits.size(); ++task)
            ASSERT_EQ(hits[task], task < size ? 1 : 0);
        ASSERT_EQ(calls.load(), expectedCalls);
    }
    // A single thread runs every task of a job in turn.
    gl::gpu::Phase2ProjectionWorkers single(1);
    std::vector<uint32_t> order;
    auto record = [&order](uint32_t task) { order.push_back(task); };
    single.run(5, record);
    ASSERT_EQ(order.size(), std::size_t(5));
    for (uint32_t task = 0; task < 5; ++task) ASSERT_EQ(order[task], task);
}

TEST(phase2_projection, complete_usage_counts_every_semantic_column) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.incubator_mode = true;
    analyzer.parameters.ban_disintegration = false;
    analyzer.parameters.compressor_mode = true;
    analyzer.parameters.standardMaxSecondaryNumber = 13;
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["theta"] = gl::LogicalEntity(
        "atomic", {}, "theta", 1);
    gl::Memory body;
    ((body).level) = 7;
    const gl::NameId mainId = ((body).nameMap).encode("main");
    const gl::NameId nameId = ((body).nameMap).encode("theta");
    body.setExprKey("gpu_test");
    const gl::IntEncodedExpr statement =
        projectionStatement(nameId, nameId, mainId, nameId);
    ((body).intEncodedStatements).push_back(statement);
    ((body).intLocalEncodedStatements).push_back(statement);
    ((body).intLocalEncodedStatementsDelta).push_back(statement);
    ((body).intExternalStatements).push_back(statement);

    const int32_t ruleHeadId = ((body).ruleInterner).encode("(theta[1])");
    const int32_t ruleSourceId =
        ((body).ruleInterner).encode("(>[theta,(theta[1])])");
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

    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).overallHashMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((body).overallHashMemory)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    ((((body).overallHashMemory)).encodedMap).assignRun(
        normKey, std::vector<gl::LocalMemoryValue>{ value });
    ((((body).overallHashMemory)).remainingArgsNormalizedEncodedMap).assignRun(
        remainingKey, std::vector<gl::NormKey>{ normKey });
    ((((body).overallHashMemory)).productsOfRecursionIds).mint(nameId);
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).localHashMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).localHashMemoryDelta)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).workingMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);

    const gl::StatementKey statementKey{ nameId, mainId };
    const int64_t packedStatementKey = gl::packStatementKey(nameId, mainId);
    ((body).intKnownStatements).upsert(
        statementKey, gl::StatementFlags{ true, false });
    ((body).intLocalEncodedStatementsSet).mint(packedStatementKey);
    const int levels[2] = { 2, 5 };
    ((body).intStatementLevelsMap).assignSet(statementKey, levels, 2);
    ((body).intValidityNamesToFilter).mint(mainId);
    ((body).frozenOrBranches).mint(nameId);
    const int goalAux[1] = { 7 };
    ((body).intToBeProved).assignSet(statementKey, goalAux, 1);
    gl::tests::admitEveryStatementShape(body);

    const gl::gpu::Phase2ProjectionUsage usage =
        gl::gpu::measurePhase2ProjectionUsage(body, analyzer.anchorInfo);
    ASSERT_EQ(usage.logicalBlocks, 1u);
    ASSERT_EQ(usage.statements, 1u);
    ASSERT_EQ(usage.nameSlots, 4u);
    ASSERT_EQ(usage.ruleStringRecords, 3u);
    ASSERT_EQ(usage.ruleStringBytes,
        static_cast<uint64_t>(((body).ruleInterner).decodeView(ruleHeadId).len
            + ((body).ruleInterner).decodeView(ruleSourceId).len));
    ASSERT_EQ(usage.byteMapViews, 10u);
    ASSERT_EQ(usage.byteMapEntries, 7u);
    ASSERT_EQ(usage.byteMapSlots, 14u);
    const uint64_t normKeyBytes =
        static_cast<uint64_t>(gl::Codec<gl::NormKey>::encode(normKey).size());
    const uint64_t remainingKeyBytes =
        static_cast<uint64_t>(gl::Codec<gl::Int16SetKey>::encode(
            remainingKey).size());
    ASSERT_EQ(usage.byteKeyBytes, normKeyBytes + remainingKeyBytes);
    ASSERT_EQ(usage.blobRecords, 3u);
    ASSERT_EQ(usage.blobBytes,
        static_cast<uint64_t>(gl::Codec<gl::OwnerSet>::serialize(owner).size()
            + gl::Codec<gl::LocalMemoryValue>::serialize(value).size()
            + gl::Codec<gl::NormKey>::serialize(normKey).size()));
    ASSERT_EQ(usage.reverseMapViews, 1u);
    ASSERT_EQ(usage.reverseMapEntries, 1u);
    ASSERT_EQ(usage.reverseMapSlots, 2u);
    ASSERT_EQ(usage.reverseKeyBytes, 0u);
    ASSERT_EQ(usage.reverseOwners, 1u);
    ASSERT_EQ(usage.podMapViews, 9u);
    ASSERT_EQ(usage.podMapEntries, 9u);
    ASSERT_EQ(usage.podMapSlots, 18u);
    ASSERT_EQ(usage.podRunValues, 3u);
    ASSERT_EQ(usage.mandatoryStatementKeys, 3u);
    ASSERT_EQ(usage.metadataBytes,
        8u + static_cast<uint64_t>(analyzer.anchorInfo.name.size()));

    gl::gpu::Phase2ProjectionArena arena(projectionCapacity(
        1, 1, static_cast<uint32_t>(((body).nameMap).nameCount() + 1), 256));
    const gl::gpu::DeviceLogicalBlockProjection projected =
        arena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(projected.byteMapViewCount, 10u);
    ASSERT_EQ(projected.reverseMapViewCount, 1u);
    ASSERT_EQ(projected.podMapViewCount, 9u);
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
    ASSERT_EQ(((projected).level), 7);
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
        arena.blobBytes.data() + reverseEntry.keyOffset,
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
            body, analyzer.anchorInfo, localOnly);
    ASSERT_EQ(selectedUsage.byteMapViews, 10u);
    ASSERT_LT(selectedUsage.byteMapEntries, usage.byteMapEntries);
    ASSERT_LT(selectedUsage.byteMapSlots, usage.byteMapSlots);
    ASSERT_EQ(selectedUsage.byteKeyBytes, usage.byteKeyBytes);

    gl::gpu::Phase2ProjectionArena selectedArena(projectionCapacity(
        1, 1, static_cast<uint32_t>(((body).nameMap).nameCount() + 1), 256));
    const gl::gpu::DeviceLogicalBlockProjection selected =
        selectedArena.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, localOnly);
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

// The CUDA packing reads the measurement each logic block records at the end
// of its phase 1: the stump-free usage plus a part's stumps equals the direct
// measurement, the selection names the batches' hash memories, and the record
// stores exactly those values with the burst stamp.
TEST(phase2_projection, recorded_phase2_measurement_equals_the_direct_one) {
    const auto bit = [](gl::gpu::DeviceHashMemoryKind memory) {
        return 1u << static_cast<uint32_t>(memory);
    };
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    const gl::NameId main = ((body).nameMap).encode("main");
    const gl::NameId name = ((body).nameMap).encode("recorded-shape");
    const gl::IntEncodedExpr statement =
        projectionStatement(name, name, main, name);
    ((body).intLocalEncodedStatements).push_back(statement);
    ((body).intExternalStatements).push_back(statement);
    const gl::NormKey key{ 1, { name } };
    gl::LocalMemoryValue value;
    value.valueId = ((body).ruleInterner).encode("recorded-rule");
    ((((body).workingMemory)).encodedMap).assignRun(
        key, std::vector<gl::LocalMemoryValue>{ value });
    ((((body).localHashMemory)).encodedMap).assignRun(
        key, std::vector<gl::LocalMemoryValue>{ value });

    // Selection: working (map + local statements), overall (externals with
    // local statements) and local (map + externals); no local delta.
    const uint32_t expected = bit(gl::gpu::DeviceHashMemoryKind::working)
        | bit(gl::gpu::DeviceHashMemoryKind::overall)
        | bit(gl::gpu::DeviceHashMemoryKind::local);
    ASSERT_EQ(gl::gpu::selectPhase2HashMemories(body, false), expected);
    ASSERT_EQ(gl::gpu::selectPhase2HashMemories(body, true),
              bit(gl::gpu::DeviceHashMemoryKind::overall));
    const gl::Memory emptyBody;
    ASSERT_EQ(gl::gpu::selectPhase2HashMemories(emptyBody, false), 0u);

    // Stumps are added per part; unscheduled and counter-example usage keep none.
    const gl::gpu::Phase2TaskProjectionUsage stumpFree =
        gl::gpu::measurePhase2TaskProjectionUsage(body, 0, false);
    const gl::gpu::Phase2TaskProjectionUsage withStumps =
        gl::gpu::phase2TaskUsageWithStumps(stumpFree, 5, false);
    const gl::gpu::Phase2TaskProjectionUsage direct =
        gl::gpu::measurePhase2TaskProjectionUsage(body, 5, false);
    ASSERT_EQ(withStumps.tasks, direct.tasks);
    ASSERT_EQ(withStumps.batches, direct.batches);
    ASSERT_EQ(withStumps.terms, direct.terms);
    ASSERT_EQ(withStumps.stumps, 5u);
    const gl::gpu::Phase2TaskProjectionUsage unscheduled =
        gl::gpu::phase2TaskUsageWithStumps(
            gl::gpu::measurePhase2TaskProjectionUsage(emptyBody, 0, false), 3, false);
    ASSERT_EQ(unscheduled.tasks, 0u);
    ASSERT_EQ(unscheduled.stumps, 0u);
    const gl::gpu::Phase2TaskProjectionUsage counterExample =
        gl::gpu::phase2TaskUsageWithStumps(
            gl::gpu::measurePhase2TaskProjectionUsage(emptyBody, 0, true), 0, true);
    ASSERT_EQ(counterExample.tasks, 1u);
    ASSERT_EQ(counterExample.stumps, 0u);

    // The record holds exactly the direct values and the burst stamp.
    ((analyzer).phase2MeasurementEpoch) = 7;
    analyzer.phase2Executor.recordPhase2Measurement(body);
    ASSERT_EQ(((body).phase2MeasurementEpoch), uint64_t(7));
    ASSERT_EQ(((body).phase2TaskUsage).tasks, stumpFree.tasks);
    ASSERT_EQ(((body).phase2TaskUsage).batches, stumpFree.batches);
    ASSERT_EQ(((body).phase2TaskUsage).terms, stumpFree.terms);
    ASSERT_EQ(((body).phase2TaskUsage).stumps, 0u);
    ASSERT_EQ(((body).phase2SelectedHashMemories), expected);
    const gl::gpu::Phase2ProjectionUsage projection =
        gl::gpu::measurePhase2ProjectionUsage(body, analyzer.anchorInfo, expected, true);
    ASSERT_TRUE(std::memcmp(&((body).phase2ProjectionUsage), &projection,
                            sizeof(projection)) == 0);
    ASSERT_EQ(((body).phase2ProjectionUsage).logicalBlocks, uint64_t(1));
    gl::Memory quietBody;
    analyzer.phase2Executor.recordPhase2Measurement(quietBody);
    ASSERT_EQ(((quietBody).phase2MeasurementEpoch), uint64_t(7));
    ASSERT_EQ(((quietBody).phase2TaskUsage).tasks, 0u);
    ASSERT_EQ(((quietBody).phase2SelectedHashMemories), 0u);
    ASSERT_EQ(((quietBody).phase2ProjectionUsage).logicalBlocks, uint64_t(0));
}

TEST(phase2_projection, task_inputs_are_pointer_free_ordered_and_reusable) {
    gl::Memory measuredBody;
    const gl::NameId measuredMain = ((measuredBody).nameMap).encode("main");
    const gl::NameId measuredName = ((measuredBody).nameMap).encode("task-shape");
    const gl::IntEncodedExpr measuredStatement = projectionStatement(
        measuredName, measuredName, measuredMain, measuredName);
    ((measuredBody).intLocalEncodedStatements).push_back(measuredStatement);
    ((measuredBody).intExternalStatements).push_back(measuredStatement);
    const gl::NormKey measuredKey{ 1, { measuredName } };
    gl::LocalMemoryValue measuredValue;
    measuredValue.valueId = ((measuredBody).ruleInterner).encode("task-rule");
    ((((measuredBody).workingMemory)).encodedMap).assignRun(
        measuredKey, std::vector<gl::LocalMemoryValue>{ measuredValue });
    ((((measuredBody).localHashMemory)).encodedMap).assignRun(
        measuredKey, std::vector<gl::LocalMemoryValue>{ measuredValue });
    ((((measuredBody).localHashMemoryDelta)).encodedMap).assignRun(
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
    capacity.examinedRows = 256;
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

    schedule.appendCall(
        3, gl::gpu::DeviceHashMemoryKind::overall, 12, 1, 17, 4096);

    const gl::gpu::Phase2FilterClassCensus census =
        schedule.measureClassReuse();
    ASSERT_EQ(census.callCount, 8u);
    ASSERT_EQ(census.uniqueClassCount, 7u);
    ASSERT_EQ(census.duplicateCallCount, 1u);
    ASSERT_EQ(census.uniqueExaminedRows, 118u);
    ASSERT_EQ(census.duplicateExaminedRows, 17u);
    ASSERT_EQ(census.maximumClassMultiplicity, 2u);
    ASSERT_EQ(schedule.calls.size(), 8u);
    ASSERT_EQ(schedule.examinedRows, 135u);
    ASSERT_EQ(schedule.classes.size(), 7u);
    ASSERT_EQ(schedule.callClassIndices.size(), 8u);
    ASSERT_EQ(schedule.callClassIndices[0], 0u);
    ASSERT_EQ(schedule.callClassIndices[1], 0u);
    ASSERT_EQ(schedule.callClassIndices[2], 1u);
    ASSERT_EQ(schedule.callClassIndices[3], 2u);
    ASSERT_EQ(schedule.callClassIndices[4], 3u);
    ASSERT_EQ(schedule.callClassIndices[5], 4u);
    ASSERT_EQ(schedule.callClassIndices[6], 5u);
    ASSERT_EQ(schedule.callClassIndices[7], 6u);
    ASSERT_EQ(schedule.calls[7].maximumFilteredStatements, 4096u);
    ASSERT_EQ(schedule.classMultiplicities[0], 2u);
    for (uint32_t classIndex = 1; classIndex < 7; ++classIndex) {
        ASSERT_EQ(schedule.classMultiplicities[classIndex], 1u);
    }
    ASSERT_EQ(schedule.classExaminedRows, 118u);

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
CUDA_TEST(phase2_projection, fixed_cuda_task_buffer_upload_is_exact) {
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

CUDA_TEST(phase2_projection, filter_examined_volume_exceeds_32_bits_without_repeated_storage) {
    gl::gpu::Phase2FilterScheduleArena schedule({4096, 1ull << 34, 1, 1u << 22});
    for (uint32_t call = 0; call < 4096; ++call)
        schedule.appendCall(0, gl::gpu::DeviceHashMemoryKind::overall, 0, 1, 1u << 22);
    const auto census = schedule.measureClassReuse();
    ASSERT_EQ(schedule.examinedRows, 1ull << 34);
    ASSERT_EQ(census.uniqueClassCount, 1u);
    ASSERT_EQ(census.uniqueExaminedRows, 1ull << 22);
    ASSERT_EQ(census.duplicateExaminedRows, (1ull << 34) - (1ull << 22));
}

CUDA_TEST(phase2_projection, bulk_cuda_filter_and_sort_matches_processor_streams) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.maxIterationNumberVariable = 3;
    gl::Memory body;
    gl::NameMap& names = ((body).nameMap);
    const gl::NameId mainId = names.encode("main");
    const gl::NameId frozenBranch = names.encodePush(
        mainId, gl::StrSpan("ordis_(filter-test)", 19));
    const int32_t fixtureRuleId =
        ((body).ruleInterner).encode("(fixture-rule[1])");
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
        ((body).intEncodedStatements).push_back(statement);
    }
    ((body).frozenOrBranches).mint(frozenBranch);
    // The rejected prefix below repeats row 5, so these six rows' patterns
    // cover every row of the rebuilt list.
    gl::tests::admitEveryStatementShape(body);

    gl::HashMemory* memories[4] = {
        &((body).overallHashMemory), &((body).localHashMemory),
        &((body).localHashMemoryDelta), &((body).workingMemory)
    };
    for (uint32_t statementIndex = 0; statementIndex < 6; ++statementIndex) {
        const gl::IntEncodedExpr& statement =
            ((body).intEncodedStatements)[statementIndex];
        gl::NameId keyData[4] = {
            statement.nameId, statement.negation, 1, 0
        };
        const gl::NormKey key{
            1, std::vector<gl::NameId>(keyData, keyData + 4)
        };
        for (gl::HashMemory* memory : memories) {
            if (statementIndex == 3) {
                gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
                    ((*(memory)).normalizedEncodedKeys), 1, keyData, 4,
                    fixtureOwner, ownerArena);
            }
            else {
                gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
                    ((*(memory)).normalizedEncodedSubkeys), 1, keyData, 4);
            }
        }
    }
    // Rejected rows force retained statement indices beyond the old packed field.
    constexpr uint32_t rejectedPrefix = (1u << 18) + 1;
    gl::IntEncodedExpr acceptedStatements[6]{};
    for (uint32_t row = 0; row < 6; ++row)
        acceptedStatements[row] = ((body).intEncodedStatements)[row];
    ((body).intEncodedStatements).clear();
    gl::IntEncodedExpr rejected = acceptedStatements[5];
    for (uint32_t row = 0; row < rejectedPrefix; ++row)
        ((body).intEncodedStatements).push_back(rejected);
    for (const auto& statement : acceptedStatements)
        ((body).intEncodedStatements).push_back(statement);
    std::clog << "[phase2-filter-twin] fixture ready\n" << std::flush;

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= names.nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(names.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity projectionLimits =
        projectionCapacity(
            1, static_cast<uint32_t>(((body).intEncodedStatements).size()),
            static_cast<uint32_t>(names.nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena hostProjection(projectionLimits);
    const gl::gpu::DeviceLogicalBlockProjection block =
        hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    std::clog << "[phase2-filter-twin] projection packed\n" << std::flush;
    gl::gpu::CudaPhase2ProjectionBuffer deviceProjection(projectionLimits);
    // A uniform rank translation preserves lexical order while crossing 18 bits.
    for (auto& name : hostProjection.nameRecords)
        name.decodedLexRank += (1u << 18);
    deviceProjection.upload(hostProjection);
    std::clog << "[phase2-filter-twin] projection uploaded\n" << std::flush;

    const auto processorOrder = [&](const char* label,
                                    gl::HashMemory& memory,
                                    bool alsoAcceptFullKeys) {
        std::clog << "[phase2-filter-twin] processor " << label
                  << " start\n" << std::flush;
        gl::NameId output[8192]{};
        const gl::NameId count = gl::tests::HashRequestGeneratorFixture::filterIntEncodedStatements(analyzer.requestGenerator,
            gl::IntStmtView(((body).intEncodedStatements)), memory, body,
            alsoAcceptFullKeys, output, 8192);
        std::vector<gl::NameId> result(output, output + count);
        std::stable_sort(result.begin(), result.end(),
            [&](gl::NameId left, gl::NameId right) {
                return gl::compareSpans(
                    names.decodeView(((body).intEncodedStatements)[left].nameId),
                    names.decodeView(((body).intEncodedStatements)[right].nameId)) < 0;
            });
        std::clog << "[phase2-filter-twin] processor " << label
                  << " complete\n" << std::flush;
        return result;
    };
    const std::vector<gl::NameId> expected[5] = {
        processorOrder("overall-subkey", ((body).overallHashMemory), false),
        processorOrder("overall-union", ((body).overallHashMemory), true),
        processorOrder("local", ((body).localHashMemory), true),
        processorOrder("local-delta", ((body).localHashMemoryDelta), true),
        processorOrder("working", ((body).workingMemory), true)
    };
    std::clog << "[phase2-filter-twin] processor oracle complete\n"
              << std::flush;

    gl::gpu::Phase2FilterScheduleCapacity filterLimits{};
    filterLimits.calls = 2048;
    filterLimits.examinedRows = 16777216;
    filterLimits.retainedRows = 1048576;
    filterLimits.maximumExaminedRowsPerCall = rejectedPrefix + 6;
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
            ASSERT_TRUE((key >> gl::gpu::kDeviceFilterStatementIndexBits)
                >= (1u << 18));
            ASSERT_TRUE(expected[call][row] >= rejectedPrefix);
            ASSERT_EQ(static_cast<gl::NameId>(
                key & gl::gpu::kDeviceFilterStatementIndexMask),
                expected[call][row]);
        }
    }
    ASSERT_EQ(keyOffset, retained);
}

CUDA_TEST(phase2_projection, premise_pattern_rejections_keep_processor_and_device_lists_equal) {
    // Every row's signed core is a premise core, so the shape keys alone would
    // keep them all. The premise patterns reject three: `(p[j])` (no premise
    // carries the literal j), `(r[a])` at the sibling scope T (its only premise
    // lives at scope S) and `(s[a,a])` (the premise repeats no variable). The
    // upload omits exactly those rows, and every one of the five filter calls
    // keeps the same rows, in the same order, on the device as on the
    // processor (I-300).
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    gl::NameMap& names = ((body).nameMap);
    const gl::NameId mainId = names.encode("main");
    const gl::NameId scopeS = names.encodePush(mainId, gl::StrSpan("hypo_(h[1])", 11));
    const gl::NameId scopeT = names.encodePush(mainId, gl::StrSpan("hypo_(h[2])", 11));
    const int32_t fixtureRuleId =
        ((body).ruleInterner).encode("(fixture-rule[2])");
    const gl::RuleOwner fixtureOwner =
        gl::packRuleOwner(fixtureRuleId, mainId);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);

    struct Row { const char* text; gl::NameId scope; };
    const Row rows[8] = {
        { "(p[k])", mainId }, { "(p[j])", mainId }, { "(q[a])", mainId },
        { "(r[a])", scopeS }, { "(r[a])", scopeT }, { "(s[a,b])", mainId },
        { "(s[a,a])", mainId }, { "(p[k])", scopeS }
    };
    for (const Row& row : rows) {
        ((body).intEncodedStatements).push_back(gl::encodeExpression(
            gl::EncodedExpression(row.text, names.decode(row.scope)), names));
    }
    gl::tests::registerPremisePattern(((body).overallHashMemory),
        gl::tests::premiseRowAt(names, "(p[u_k])", mainId), mainId, 1);
    gl::tests::registerPremisePattern(((body).localHashMemory),
        gl::tests::premiseRowAt(names, "(q[x])", mainId), mainId, 2);
    gl::tests::registerPremisePattern(((body).localHashMemoryDelta),
        gl::tests::premiseRowAt(names, "(r[x])", scopeS), scopeS, 3);
    gl::tests::registerPremisePattern(((body).workingMemory),
        gl::tests::premiseRowAt(names, "(s[x,y])", mainId), mainId, 4);

    gl::HashMemory* memories[4] = {
        &((body).overallHashMemory), &((body).localHashMemory),
        &((body).localHashMemoryDelta), &((body).workingMemory)
    };
    for (uint32_t index = 0; index < 8; ++index) {
        // The row's one-element key as the filter builds it: signed core,
        // then per argument its first-appearance number and a zero flag.
        const gl::IntEncodedExpr& statement = ((body).intEncodedStatements)[index];
        gl::NameId keyData[gl::ExecutionParameters::MAX_KEY_SLOTS];
        gl::NameId len = 0;
        keyData[len++] = statement.nameId;
        keyData[len++] = statement.negation;
        gl::NameId nextNumber = 1;
        for (gl::NameId j = 0; j < statement.arity; ++j) {
            gl::NameId number = 0;
            for (gl::NameId earlier = 0; earlier < j && number == 0; ++earlier) {
                if (statement.argId[earlier] == statement.argId[j])
                    number = keyData[2 + 2 * earlier];
            }
            if (number == 0) number = nextNumber++;
            keyData[len++] = number;
            keyData[len++] = 0;
        }
        for (gl::HashMemory* memory : memories) {
            if (statement.nameId == names.lookup(gl::StrSpan("q", 1))) {
                gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
                    ((*(memory)).normalizedEncodedKeys), 1, keyData, len,
                    fixtureOwner, ownerArena);
            }
            else {
                gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
                    ((*(memory)).normalizedEncodedSubkeys), 1, keyData, len);
            }
        }
    }

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= names.nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(names.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity projectionLimits =
        projectionCapacity(1, 8, static_cast<uint32_t>(names.nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena hostProjection(projectionLimits);
    const gl::gpu::DeviceLogicalBlockProjection block =
        hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    ASSERT_EQ(block.statementCount, 5u);
    gl::gpu::CudaPhase2ProjectionBuffer deviceProjection(projectionLimits);
    deviceProjection.upload(hostProjection);

    const auto processorOrder = [&](gl::HashMemory& memory, bool alsoAcceptFullKeys) {
        gl::NameId output[16]{};
        const gl::NameId count = gl::tests::HashRequestGeneratorFixture::filterIntEncodedStatements(
            analyzer.requestGenerator, gl::IntStmtView(((body).intEncodedStatements)), memory, body,
            alsoAcceptFullKeys, output, 16);
        std::vector<gl::NameId> result(output, output + count);
        std::stable_sort(result.begin(), result.end(),
            [&](gl::NameId left, gl::NameId right) {
                return gl::compareSpans(
                    names.decodeView(((body).intEncodedStatements)[left].nameId),
                    names.decodeView(((body).intEncodedStatements)[right].nameId)) < 0;
            });
        return result;
    };
    const std::vector<gl::NameId> expected[5] = {
        processorOrder(((body).overallHashMemory), false),
        processorOrder(((body).overallHashMemory), true),
        processorOrder(((body).localHashMemory), true),
        processorOrder(((body).localHashMemoryDelta), true),
        processorOrder(((body).workingMemory), true)
    };
    // The full-key call keeps the q row, the subkey-only call does not;
    // neither keeps a pattern-rejected row.
    ASSERT_EQ(expected[0].size(), 4u);
    ASSERT_EQ(expected[1].size(), 5u);

    gl::gpu::Phase2FilterScheduleCapacity filterLimits{};
    filterLimits.calls = 8;
    filterLimits.examinedRows = 64;
    filterLimits.retainedRows = 64;
    filterLimits.maximumExaminedRowsPerCall = 8;
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
    const uint32_t retained = deviceFilter.filterAndSort(deviceProjection, schedule);
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
            ASSERT_EQ(static_cast<int64_t>(key & gl::gpu::kDeviceFilterStatementIndexMask),
                static_cast<int64_t>(hostProjection.projectedStatementIndex(0, expected[call][row])));
        }
    }
    ASSERT_EQ(keyOffset, retained);
}

CUDA_TEST(phase2_projection, configurable_filter_keeps_rows_beyond_old_limit) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.maxIterationNumberVariable = 1;
    gl::Memory body;
    const gl::NameId main = ((body).nameMap).encode("main");
    gl::IntEncodedExpr first = gl::encodeExpression(
        gl::EncodedExpression("(z[a])", "main"), ((body).nameMap));
    const gl::IntEncodedExpr second = gl::encodeExpression(
        gl::EncodedExpression("(a[b])", "main"), ((body).nameMap));
    for (const auto& row : {first, second}) {
        const gl::NameId key[] = {row.nameId, row.negation, 1, 0};
        gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
            ((((body).overallHashMemory)).normalizedEncodedSubkeys), 1, key, 4);
    }
    constexpr uint32_t rejected = 257;
    constexpr uint32_t eligible = 9040;
    first.maxIteration = 2;
    for (uint32_t i = 0; i < rejected; ++i)
        ((body).intEncodedStatements).push_back(first);
    first.maxIteration = -1;
    for (uint32_t i = 0; i < eligible; ++i)
        ((body).intEncodedStatements).push_back(i % 2 == 0 ? first : second);
    gl::tests::admitEveryStatementShape(body);

    uint32_t nameBytes = 0;
    for (gl::NameId i = 1; i <= ((body).nameMap).nameCount(); ++i)
        nameBytes += static_cast<uint32_t>(((body).nameMap).decodeView(i).len);
    const auto limits = projectionCapacity(1, rejected + eligible,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena host(limits);
    const auto block = host.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    gl::gpu::CudaPhase2ProjectionBuffer device(limits);
    device.upload(host);
    const gl::gpu::Phase2FilterScheduleCapacity filterLimits{
        3, 3ull * (rejected + eligible), 32768, rejected + eligible};
    gl::gpu::Phase2FilterScheduleArena schedule(filterLimits);
    const uint32_t caps[] = {3, 8192, 16384};
    std::vector<std::vector<gl::NameId>> expected;
    for (const uint32_t cap : caps) {
        analyzer.parameters.maxNumberFilteredStatements = static_cast<int>(cap);
        std::vector<gl::NameId> rows(cap);
        const gl::NameId count = gl::tests::HashRequestGeneratorFixture::filterIntEncodedStatements(analyzer.requestGenerator,
            gl::IntStmtView(((body).intEncodedStatements)), ((body).overallHashMemory),
            body, true, rows.data(), analyzer.parameters.maxNumberFilteredStatements);
        rows.resize(count);
        ASSERT_EQ(static_cast<uint32_t>(count), std::min(cap, eligible));
        ASSERT_EQ(rows.front(), static_cast<gl::NameId>(rejected));
        ASSERT_EQ(rows.back(), static_cast<gl::NameId>(rejected + count - 1));
        std::stable_sort(rows.begin(), rows.end(), [&](gl::NameId l, gl::NameId r) {
            return gl::compareSpans(((body).nameMap).decodeView(((body).intEncodedStatements)[l].nameId),
                ((body).nameMap).decodeView(((body).intEncodedStatements)[r].nameId)) < 0;
        });
        expected.push_back(rows);
        schedule.appendCall(0, gl::gpu::DeviceHashMemoryKind::overall,
            1, 1, block.statementCount, cap);
    }
    ASSERT_EQ(schedule.classes.size(), 3u);
    gl::gpu::CudaPhase2FilterSortBuffer filter(filterLimits);
    const uint32_t retained = filter.filterAndSort(device, schedule);
    uint32_t counts[3]{};
    ASSERT_EQ(filter.downloadCallCounts(counts, 3), 3u);
    std::vector<uint64_t> keys(retained);
    ASSERT_EQ(filter.downloadSortedKeys(keys.data(), retained), retained);
    uint32_t offset = 0;
    for (uint32_t call = 0; call < 3; ++call) {
        ASSERT_EQ(counts[call], expected[call].size());
        for (const auto index : expected[call]) {
            ASSERT_EQ(static_cast<gl::NameId>(keys[offset++]
                & gl::gpu::kDeviceFilterStatementIndexMask), index);
        }
    }
    ASSERT_EQ(offset, retained);
    ASSERT_EQ(counts[2], eligible);
}

CUDA_TEST(phase2_projection, fixed_cuda_buffer_upload_and_lookups_are_exact) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["delta"] = gl::LogicalEntity(
        "atomic", {}, "delta", 1);
    gl::Memory body;
    const gl::NameId mainId = ((body).nameMap).encode("main");
    const gl::NameId nameId = ((body).nameMap).encode("delta");
    const gl::NameId originalId = ((body).nameMap).encode("(delta[delta])");
    const gl::IntEncodedExpr statement =
        projectionStatement(nameId, originalId, mainId, nameId);
    ((body).intEncodedStatements).push_back(statement);
    ((body).intLocalEncodedStatements).push_back(statement);
    ((body).intLocalEncodedStatementsDelta).push_back(statement);
    ((body).intExternalStatements).push_back(statement);
    body.setExprKey("cuda_projection");

    const int32_t ruleId = ((body).ruleInterner).encode("(delta[1])");
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
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).overallHashMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((body).overallHashMemory)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    ((((body).overallHashMemory)).encodedMap).assignRun(
        normKey, std::vector<gl::LocalMemoryValue>{ value });
    ((((body).overallHashMemory)).remainingArgsNormalizedEncodedMap).assignRun(
        remainingKey, std::vector<gl::NormKey>{ normKey });
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).localHashMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((body).localHashMemory)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).localHashMemoryDelta)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((body).localHashMemoryDelta)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).workingMemory)).normalizedEncodedKeys),
        normKey.numberExpressions, normKey.data.data(),
        static_cast<int32_t>(normKey.data.size()), packedOwner, ownerArena);
    ((((body).workingMemory)).normalizedEncodedSubkeys).assignRun(
        normKey, std::vector<gl::OwnerSet>{ owner });
    ((((body).overallHashMemory)).productsOfRecursionIds).mint(nameId);
    const gl::StatementKey statementKey{ originalId, mainId };
    const int64_t packedStatementKey =
        gl::packStatementKey(originalId, mainId);
    ((body).intKnownStatements).upsert(
        statementKey, gl::StatementFlags{ true, true });
    ((body).intLocalEncodedStatementsSet).mint(packedStatementKey);
    const int levels[1] = { 4 };
    ((body).intStatementLevelsMap).assignSet(statementKey, levels, 1);
    ((body).intValidityNamesToFilter).mint(mainId);
    ((body).frozenOrBranches).mint(nameId);
    ((body).intToBeProved).assignSet(statementKey, levels, 1);
    gl::tests::admitEveryStatementShape(body);

    uint32_t byteCount = 0;
    for (gl::NameId id = 1; id <= ((body).nameMap).nameCount(); ++id)
        byteCount += static_cast<uint32_t>(((body).nameMap).decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity capacity = projectionCapacity(
        1,
        1,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1),
        byteCount);
    gl::gpu::Phase2ProjectionArena host(capacity);
    const gl::gpu::DeviceLogicalBlockProjection projected =
        host.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
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

    // Compare two complete streamed windows with the existing in-place
    // shard-adoption oracle, including every nonempty semantic column.
    const auto twice = gl::gpu::addProjectionCapacity(capacity, capacity);
    gl::gpu::Phase2ProjectionArena expected(twice, twice);
    gl::gpu::Phase2ProjectionArena first(capacity, expected, {});
    gl::gpu::Phase2ProjectionArena second(capacity, expected, capacity);
    first.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    second.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    gl::gpu::Phase2ProjectionArena* oracleShards[] = {&first, &second};
    uint32_t firstBlocks[2]{};
    gl::gpu::Phase2ProjectionWorkers oracleWorkers(2);
    expected.adoptShards(oracleShards, 2, firstBlocks,
        gl::gpu::Phase2ProjectionArena::ShardBases{}, oracleWorkers);
    const auto backingLayout = gl::gpu::Phase2ProjectionArena::hostLayout(
        twice, twice, gl::gpu::Phase2ProjectionArena::Role::Backing);
    ASSERT_EQ(backingLayout.roundedBytes(gl::gpuHostStorage().blockBytes()), expected.hostBytes());

    gl::gpu::Phase2ProjectionArena retained(twice,
        gl::gpu::Phase2ProjectionArena::SealingOnly{});
    const uint64_t retainedBytes = retained.hostBytes();
    const auto retainedLayout = gl::gpu::Phase2ProjectionArena::hostLayout(
        twice, twice, gl::gpu::Phase2ProjectionArena::Role::Sealing);
    ASSERT_TRUE(retainedLayout.bytes < backingLayout.bytes);
    ASSERT_EQ(retainedBytes, retainedLayout.roundedBytes(gl::gpuHostStorage().blockBytes()));
    ASSERT_TRUE(retainedBytes <= expected.hostBytes());
    ASSERT_EQ(retained.statements.capacity(), std::size_t(0));
    ASSERT_EQ(retained.byteKeyBytes.capacity(), std::size_t(0));
    gl::gpu::CudaPhase2ProjectionBuffer streamed(twice);
    gl::gpu::Phase2ProjectionArena::ShardBases ends{};
    streamed.beginStreamedUpload();
    for (int window = 0; window < 2; ++window) {
        gl::gpu::Phase2ProjectionArena source(capacity);
        source.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
        source.timing.mergeNanoseconds = static_cast<uint64_t>(window + 1);
        const auto next = source.streamedEnd(ends);
        source.rebaseForStream(ends);
        streamed.appendStreamedUpload(source, ends);
        retained.appendSealingProjection(source, ends);
        ASSERT_EQ(retained.timing.logicalBlocks, retained.logicalBlocks.size());
        ASSERT_EQ(retained.timing.logicalBlocks, static_cast<uint64_t>(window + 1));
        ends = next;
    }
    streamed.finishPhase2Upload();
    ASSERT_EQ(streamed.launchChecksum(), projectionChecksum(expected));
    ASSERT_EQ(retained.timing.mergeNanoseconds, 3ull);
    ASSERT_EQ(retained.timing.sourceStatementRows, expected.timing.sourceStatementRows);
    ASSERT_EQ(retained.timing.retainedStatementRows, expected.timing.retainedStatementRows);
    ASSERT_EQ(retained.timing.prefixFilterKeys, expected.timing.prefixFilterKeys);
    ASSERT_EQ(retained.timing.prefixFilterPredicates, expected.timing.prefixFilterPredicates);
    ASSERT_EQ(retained.timing.prefixFilterBytes, expected.timing.prefixFilterBytes);
    ASSERT_EQ(retained.logicalBlocks.size(), expected.logicalBlocks.size());
    ASSERT_EQ(std::memcmp(retained.logicalBlocks.data(), expected.logicalBlocks.data(),
        expected.logicalBlocks.size() * sizeof(gl::gpu::DeviceLogicalBlockProjection)), 0);
    ASSERT_TRUE(retained.sourceStatementIndices == expected.sourceStatementIndices);
    ASSERT_TRUE(retained.nameBytes == expected.nameBytes);
    ASSERT_TRUE(retained.ruleStringBytes == expected.ruleStringBytes);
    ASSERT_EQ(std::memcmp(retained.nameRecords.data(), expected.nameRecords.data(),
        expected.nameRecords.size() * sizeof(gl::gpu::DeviceNameRecord)), 0);
    ASSERT_EQ(std::memcmp(retained.ruleStringRecords.data(), expected.ruleStringRecords.data(),
        expected.ruleStringRecords.size() * sizeof(gl::gpu::DeviceRuleStringRecord)), 0);
    for (uint32_t block = 0; block < retained.logicalBlocks.size(); ++block) {
        const auto& descriptor = retained.logicalBlocks[block];
        for (uint32_t row = 0; row < descriptor.statementCount; ++row) {
            const auto sourceIndex = retained.sourceStatementIndices[descriptor.statementOffset + row];
            ASSERT_EQ(retained.projectedStatementIndex(block, sourceIndex), static_cast<gl::NameId>(row));
        }
    }
    // The upload thread reproduces the synchronous sequence: windows adopted at
    // their stream bases are uploaded and retained in submission order while
    // the next window is built (D-447), and each window's slot is emptied as
    // soon as its synchronous copies finish
    // (D-491).
    for (std::size_t slotCount : {std::size_t(2), std::size_t(1)}) {
        gl::gpu::Phase2ProjectionWorkers workers(2);
        gl::gpu::CudaPhase2ProjectionBuffer overlapped(twice);
        gl::gpu::Phase2ProjectionArena overlappedRetained(twice,
            gl::gpu::Phase2ProjectionArena::SealingOnly{});
        overlapped.beginStreamedUpload();
        gl::gpu::Phase2WindowUploader uploader(overlapped, overlappedRetained);
        uploader.quiesceForSnapshot();
        ASSERT_EQ(uploader.observeForSnapshot(),uint64_t(0));
        gl::gpu::Phase2ProjectionArena::ShardBases windowEnds{};
        gl::GpuHostObjectArray<std::optional<gl::gpu::Phase2ProjectionArena>> windows(slotCount);
        for (int windowIndex = 0; windowIndex < 2; ++windowIndex) {
            if (slotCount == 1) uploader.waitUntilUploaded();
            auto& slot = windows[windowIndex % slotCount];
            ASSERT_FALSE(slot.has_value());
            slot.emplace(capacity, capacity);
            auto* window = &*slot;
            uint32_t windowFirst[1]{ 3 };
            {
                gl::gpu::Phase2ProjectionArena shard(
                    capacity, *window, gl::gpu::Phase2ProjectionCapacity{});
                shard.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
                gl::gpu::Phase2ProjectionArena* windowShards[] = { &shard };
                window->adoptShards(windowShards, 1, windowFirst, windowEnds, workers);
            }
            ASSERT_EQ(windowFirst[0], 0u);
            const auto next = window->streamedEnd(windowEnds);
            uploader.submit(slot, windowEnds);
            windowEnds = next;
        }
        // An idle uploader has already released the last window.
        uploader.quiesceForSnapshot();
        ASSERT_EQ(uploader.observeForSnapshot(), uint64_t(0));
        for (std::size_t slot = 0; slot < slotCount; ++slot)
            ASSERT_FALSE(windows[slot].has_value());
        uploader.waitUntilUploaded();
        ASSERT_EQ(uploader.observeForSnapshot(), uint64_t(0));
        uploader.finish();
        overlapped.finishPhase2Upload();
        ASSERT_EQ(overlapped.launchChecksum(), projectionChecksum(expected));
        ASSERT_EQ(overlapped.pinnedHostBytes(), uint64_t(0));
        ASSERT_EQ(overlappedRetained.logicalBlocks.size(), retained.logicalBlocks.size());
        ASSERT_EQ(std::memcmp(overlappedRetained.logicalBlocks.data(), retained.logicalBlocks.data(),
            retained.logicalBlocks.size() * sizeof(gl::gpu::DeviceLogicalBlockProjection)), 0);
        ASSERT_TRUE(overlappedRetained.sourceStatementIndices == retained.sourceStatementIndices);
        ASSERT_TRUE(overlappedRetained.nameBytes == retained.nameBytes);
        ASSERT_TRUE(overlappedRetained.ruleStringBytes == retained.ruleStringBytes);
        ASSERT_EQ(std::memcmp(overlappedRetained.nameRecords.data(), retained.nameRecords.data(),
            retained.nameRecords.size() * sizeof(gl::gpu::DeviceNameRecord)), 0);
        ASSERT_EQ(std::memcmp(overlappedRetained.ruleStringRecords.data(),
            retained.ruleStringRecords.data(),
            retained.ruleStringRecords.size() * sizeof(gl::gpu::DeviceRuleStringRecord)), 0);
        ASSERT_EQ(overlappedRetained.timing.logicalBlocks, uint64_t(2));
    }
    retained.clear();
    ASSERT_TRUE(retained.logicalBlocks.empty());
    ASSERT_TRUE(retained.nameBytes.empty());
    ASSERT_EQ(retained.hostBytes(), retainedBytes);


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
    ASSERT_EQ(nameHit.firstIntValue, ((body).nameMap).parentOf(nameId));
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
             gl::gpu::DevicePodMapKind::goals);
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

CUDA_TEST(phase2_projection, cuda_normalized_key_and_owner_gate_match_processor) {
    gl::Memory body;
    body.setExprKey("cuda_growth_candidate");
    gl::tests::ProverFixture analyzer("Peano");
    const gl::NameId mainId = ((body).nameMap).encode("main");
    const gl::NameId alphaId = ((body).nameMap).encode("alpha");
    const gl::NameId betaId = ((body).nameMap).encode("beta");
    const gl::NameId gammaId = ((body).nameMap).encode("gamma");
    const gl::NameId patternX = ((body).nameMap).encode("x");
    const gl::NameId patternY = ((body).nameMap).encode("y");
    const gl::NameId literalA = ((body).nameMap).encode("u_literal_a");
    const gl::NameId literalB = ((body).nameMap).encode("u_literal_b");
    const gl::NameId wrongLiteral = ((body).nameMap).encode("u_wrong");

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
    ((body).intEncodedStatements).push_back(first);
    ((body).intEncodedStatements).push_back(second);
    ((body).intEncodedStatements).push_back(third);
    gl::tests::admitEveryStatementShape(body);

    const gl::IntEncodedExpr* expressions[3] = {
        &((body).intEncodedStatements)[0],
        &((body).intEncodedStatements)[1],
        &((body).intEncodedStatements)[2]
    };
    gl::NameId expectedData[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
    const gl::NameId expectedLength = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(
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
    const int32_t acceptingRuleId = ((body).ruleInterner).encode(
        "(cuda-accepting-rule[1])");
    const int32_t rejectingRuleId = ((body).ruleInterner).encode(
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

    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).overallHashMemory)).normalizedEncodedKeys),
        key.numberExpressions, key.data.data(),
        static_cast<int32_t>(key.data.size()), acceptingRuleOwner, ownerArena);
    ((((body).overallHashMemory)).normalizedEncodedSubkeys).assignRun(
        key, std::vector<gl::OwnerSet>{ acceptingOwner });
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).localHashMemory)).normalizedEncodedKeys),
        key.numberExpressions, key.data.data(),
        static_cast<int32_t>(key.data.size()), rejectingRuleOwner, ownerArena);
    ((((body).localHashMemory)).normalizedEncodedSubkeys).assignRun(
        key, std::vector<gl::OwnerSet>{ rejectingOwner });

    gl::NameId shortData[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
    const gl::NameId shortLength = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(
        expressions, 2, shortData,
        gl::ExecutionParameters::MAX_KEY_SLOTS);
    const gl::NormKey shortKey{
        2, std::vector<gl::NameId>(shortData, shortData + shortLength) };
    ((((body).localHashMemoryDelta)).normalizedEncodedSubkeys).assignRun(
        shortKey, std::vector<gl::OwnerSet>{ rejectingOwner });

    // A production install indexes each singleton premise as well as the
    // multi-premise key. Keep this direct key-probe fixture complete.
    for (const gl::IntEncodedExpr* expression : expressions) {
        gl::NameId singleton[gl::ExecutionParameters::MAX_KEY_SLOTS];
        const gl::NameId length = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(
            &expression, 1, singleton, gl::ExecutionParameters::MAX_KEY_SLOTS);
        gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
            ((((body).overallHashMemory)).normalizedEncodedSubkeys), 1,
            singleton, length);
    }

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= ((body).nameMap).nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(((body).nameMap).decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity capacity = projectionCapacity(
        1, static_cast<uint32_t>(((body).intEncodedStatements).size()),
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena host(capacity);
    host.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
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

CUDA_TEST(phase2_projection, cuda_bulk_growth_matches_request_gate_content) {
    gl::tests::RequestFrontierTestFiles requestFiles("request_frontier_cuda_tests");
    const auto diskStorage = requestFiles.options(1);
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.maxIterationNumberVariable = 3;
    analyzer.parameters.standardMaxSecondaryNumber = 9;
    analyzer.parameters.maxLenHypoKey = 3;
    analyzer.parameters.maxNumberSecondaryVariables = 1;
    analyzer.parameters.maxNumberSecondaryVariablesOrint = 1;
    analyzer.parameters.incubator_mode = true;
    analyzer.parameters.ban_disintegration = false;
    analyzer.parameters.compressor_mode = false;
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["gpuHead"] = gl::LogicalEntity(
        "atomic", {}, "gpuHead", 2);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["gpuMarker"] = gl::LogicalEntity(
        "expression", {}, "gpuMarker", 2);
    gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["c"] = gl::LogicalEntity(
        "expression", {}, "c", 1);

    gl::Memory body;
    ((body).level) = 1;
    body.setExprKey("cuda_bulk_growth");
    gl::NameMap& names = ((body).nameMap);
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
            ? "(c[int_0_lev_1_2])"
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
        ((body).intEncodedStatements).push_back(statement);
    }
    gl::tests::admitEveryStatementShape(body);
    ((((body).overallHashMemory)).productsOfRecursionIds).mint(variableProduct);
    ((((body).overallHashMemory)).productsOfRecursionIds).mint(variableV);
    const gl::IntEncodedExpr& mandatory = ((body).intEncodedStatements)[2];
    ((body).intLocalEncodedStatementsSet).mint(gl::packStatementKey(
        mandatory.originalId, mandatory.validityId));

    const auto normalizedKey = [&](const std::vector<gl::NameId>& indices) {
        const gl::IntEncodedExpr* expressions[
            gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        for (std::size_t index = 0; index < indices.size(); ++index)
            expressions[index] = &((body).intEncodedStatements)[indices[index]];
        gl::NameId data[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        const gl::NameId length = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(
            expressions, static_cast<gl::NameId>(indices.size()), data,
            gl::ExecutionParameters::MAX_KEY_SLOTS);
        return gl::NormKey{
            static_cast<gl::NameId>(indices.size()),
            std::vector<gl::NameId>(data, data + length) };
    };
    const int32_t fixtureRuleId = ((body).ruleInterner).encode(
        "(cuda-growth-rule[1])");
    const gl::RuleOwner fixtureOwner = gl::packRuleOwner(
        fixtureRuleId, mainId);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    const auto addLooseSubkey = [&](const std::vector<gl::NameId>& indices) {
        const gl::NormKey key = normalizedKey(indices);
        ASSERT_TRUE(key.numberExpressions < gl::HashRuleIndex::kSubkeyUCheckMinElements);
        gl::tests::HashRuleInstallerFixture::ensureEmptySubkeyRecord(
            ((((body).overallHashMemory)).normalizedEncodedSubkeys),
            key.numberExpressions, key.data.data(), static_cast<int32_t>(key.data.size()));
        const int32_t id = ((((body).overallHashMemory)).normalizedEncodedSubkeys).lookup(key);
        ASSERT_TRUE(id != 0);
        const auto value = ((((body).overallHashMemory)).normalizedEncodedSubkeys).recordAt(id, 0);
        ASSERT_FALSE(value.hasLooseOwner);
        ASSERT_TRUE(value.owners.empty() && value.uSignatures.empty());
        ASSERT_EQ(gl::Codec<gl::OwnerSet>::serialize(value).size(), 9u);
    };
    const auto addWhole = [&](const std::vector<gl::NameId>& indices) {
        const gl::NormKey key = normalizedKey(indices);
        gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
            ((((body).overallHashMemory)).normalizedEncodedKeys),
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
    ((((body).overallHashMemory)).normalizedEncodedSubkeys).assignRun(
        normalizedKey({ 0, 1, 2 }),
        std::vector<gl::OwnerSet>{ rejectingOwner });
    addWhole({ 2 });
    addWhole({ 0, 2 });
    addWhole({ 1, 2 });
    addWhole({ 0, 1, 2 });
    addWhole({ 2, 3 });
    addWhole({ 2, 4 });
    ((((body).overallHashMemory)).maxKeyLength) = 3;

    const std::vector<std::vector<gl::NameId>> evaluationRequests = {
        { 2 }, { 1, 2 }, { 0, 2 }, { 0, 1, 2 }
    };
    const gl::Int16SetKey remainingV{ { variableV } };
    const gl::Int16SetKey remainingW{ { variableW } };
    std::vector<gl::NormKey> rawEvaluationKeys;
    for (const auto& indices : evaluationRequests)
        rawEvaluationKeys.push_back(normalizedKey(indices));
    ((((body).overallHashMemory)).remainingArgsNormalizedEncodedMap).assignRun(
        remainingV, rawEvaluationKeys);
    ((((body).overallHashMemory)).remainingArgsNormalizedEncodedMap).assignRun(
        remainingW, std::vector<gl::NormKey>{ rawEvaluationKeys[0] });
    const auto mappedEvaluationKey = [&](const std::vector<gl::NameId>& indices) {
        const gl::IntEncodedExpr* expressions[
            gl::ExecutionParameters::MAX_EXPRESSIONS]{};
        for (std::size_t index = 0; index < indices.size(); ++index)
            expressions[index] = &((body).intEncodedStatements)[indices[index]];
        gl::NameId data[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        gl::NameId reverse[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        gl::NameId normalizedVariables = 0;
        const gl::NameId length = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncodedWithMap(
            expressions, static_cast<gl::NameId>(indices.size()),
            std::set<gl::NameId>{ variableV }, data,
            gl::ExecutionParameters::MAX_KEY_SLOTS,
            reverse, normalizedVariables);
        return gl::NormKey{
            static_cast<gl::NameId>(indices.size()),
            std::vector<gl::NameId>(data, data + length) };
    };
    const int32_t gpuHeadId = ((body).ruleInterner).encode(
        "(gpuHead[1,u_u_static])");
    const int32_t gpuMarkerId = ((body).ruleInterner).encode(
        "(gpuMarker[marker,1])");
    const int32_t gpuMarkerKeyAId = ((body).ruleInterner).encode(
        "(gpuKeyA[1,u_u_tail])");
    const int32_t gpuMarkerKeyBId = ((body).ruleInterner).encode(
        "(gpuKeyB[u_u_static,1])");
    const int32_t gpuRemainingZetaId = ((body).ruleInterner).encode("zeta");
    const int32_t gpuRemainingAlphaId = ((body).ruleInterner).encode("alpha");
    const int32_t gpuSourceId = ((body).ruleInterner).encode(
        "(>[gpuPremise,(gpuHead[1,u_u_static])])");
    ((body).recursionAuxyImplId) = gpuSourceId;
    // One request's rule run; the marker of request 2 carries
    // @p markerJustification.
    const auto assignEvaluationValues = [&](std::size_t index,
                                            gl::RuleJustification markerJustification) {
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
            values[1].justification = markerJustification;
            values[1].keyIds = { gpuMarkerKeyAId, gpuMarkerKeyBId };
            values[1].remainingArgIds = {
                gpuRemainingZetaId, gpuRemainingAlphaId,
                gpuRemainingAlphaId };
        }
        ((((body).overallHashMemory)).encodedMap).assignRun(
            mappedEvaluationKey(evaluationRequests[index]),
            values);
    };
    for (std::size_t index = 0; index < evaluationRequests.size(); ++index)
        assignEvaluationValues(index, gl::RuleJustification::implication);
    const gl::IntEncodedExpr& initiallyKnown = ((body).intEncodedStatements)[2];
    ((body).intKnownStatements).upsert(
        gl::StatementKey{ initiallyKnown.originalId, initiallyKnown.validityId },
        gl::StatementFlags{ true, false });
    const int initiallyKnownLevels[2] = { -1, 5 };
    ((body).intStatementLevelsMap).assignSet(
        gl::StatementKey{ initiallyKnown.originalId, initiallyKnown.validityId },
        initiallyKnownLevels, 2);
    const gl::NameId markerMailId = names.encode("(c[marker])");
    const gl::NameId knownNumericHead = names.encode("(gpuHead[1,static])");
    const gl::NameId knownProductHead = names.encode(
        "(gpuHead[product_variable,static])");
    const gl::NameId knownSecondaryHead = names.encode(
        "(gpuHead[secondary_v_shadow,static])");
    const gl::NameId negatedSecondaryHead = names.encode(
        "!(gpuHead[secondary_v_shadow,static])");
    ((body).intKnownStatements).upsert(
        gl::StatementKey{ knownNumericHead, mainId },
        gl::StatementFlags{ true, false });
    ((body).intKnownStatements).upsert(
        gl::StatementKey{ knownProductHead, mainId },
        gl::StatementFlags{ true, false });
    ((body).intKnownStatements).upsert(
        gl::StatementKey{ knownSecondaryHead, mainId },
        gl::StatementFlags{ true, false });

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= names.nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(names.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity projectionLimits =
        projectionCapacity(
            1, static_cast<uint32_t>(((body).intEncodedStatements).size()),
            static_cast<uint32_t>(names.nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena hostProjection(projectionLimits);
    const gl::gpu::DeviceLogicalBlockProjection block =
        hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
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
    growthLimits.frontierStoreRecords = 64;
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

    gl::gpu::Phase2GrowthCapacity streamLimits = growthLimits;
    streamLimits.frontierRecords = 2;
    streamLimits.frontierStoreRecords = 2;   // every larger level spills to host pages
    streamLimits.acceptedEvents = 2;
    streamLimits.rawRequests = 1;
    streamLimits.prefixPayloadValues = 12;
    streamLimits.prefixVariableValues = 4;
    streamLimits.prefixSecondaryValues = 2;
    streamLimits.candidateWindowRecords = 64;
    gl::gpu::CudaPhase2GrowthBuffer streamingGrowth(streamLimits);
    struct GrowthCapture {
        std::vector<gl::gpu::DeviceAcceptedGrowthEvent> events;
        std::vector<gl::gpu::DeviceRawGrowthRequest> requests;
    } capturedGrowth;
    const auto captureGrowth = [](void* context, const gl::gpu::CudaPhase2GrowthBuffer& buffer, bool) {
        auto& captured = *static_cast<GrowthCapture*>(context);
        gl::gpu::DeviceAcceptedGrowthEvent eventRows[2];
        gl::gpu::DeviceRawGrowthRequest requestRows[1];
        const uint32_t eventCount = buffer.downloadAcceptedEvents(eventRows, 2);
        const uint32_t requestCount = buffer.downloadRawRequests(requestRows, 1);
        captured.events.insert(captured.events.end(), eventRows, eventRows + eventCount);
        captured.requests.insert(captured.requests.end(), requestRows, requestRows + requestCount);
    };
    uint64_t streamedSubkeyCount = 0;
    const auto streamed = streamingGrowth.runRequestGrowthWindows(deviceProjection,
        deviceTasks, hostTasks, deviceFilter, growthSchedule,
        {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
         analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0},
        captureGrowth, &capturedGrowth, &streamedSubkeyCount, 1, nullptr, diskStorage);
    ASSERT_EQ(streamed.acceptedEventCount, 7ull);
    ASSERT_EQ(streamed.rawRequestCount, 4ull);
    ASSERT_EQ(streamedSubkeyCount, 5ull);
    ASSERT_TRUE(streamed.drainCount > 1);
    ASSERT_TRUE(streamed.maximumFrontierCount > streamLimits.frontierRecords);
    ASSERT_TRUE(streamed.maximumCandidateWindow > streamLimits.frontierRecords);
    ASSERT_TRUE(streamed.maximumCandidateWindow > streamLimits.rawRequests);
    ASSERT_TRUE(streamed.maximumHostBackingBytes > 0);
    ASSERT_TRUE(streamed.frontierSpills > 0);
    ASSERT_TRUE(streamed.frontierStorage.writes > 0);
    ASSERT_TRUE(streamed.frontierStorage.reads > 0);
    ASSERT_TRUE(streamed.frontierStorage.peakResidentPayloadBytes <= 2 * sizeof(gl::gpu::DeviceGrowthNode));
    ASSERT_EQ(streamed.uploadedFrontierBytes, streamed.downloadedFrontierBytes);
    std::vector<std::string> streamedEvents, streamedRequests;
    for (const auto& event : capturedGrowth.events) streamedEvents.push_back(eventText(event));
    for (const auto& request : capturedGrowth.requests) streamedRequests.push_back(eventText(request.event));
    std::sort(streamedEvents.begin(), streamedEvents.end());
    std::sort(streamedRequests.begin(), streamedRequests.end());
    ASSERT_EQ(streamedEvents, expectedEvents);
    ASSERT_EQ(streamedRequests, expectedRequests);

    auto smallWindowLimits = streamLimits;
    smallWindowLimits.candidateWindowRecords = 2;
    smallWindowLimits.prefixPayloadValues = growthLimits.prefixPayloadValues;
    smallWindowLimits.prefixVariableValues = growthLimits.prefixVariableValues;
    smallWindowLimits.prefixSecondaryValues = growthLimits.prefixSecondaryValues;
    gl::gpu::CudaPhase2GrowthBuffer smallWindowGrowth(smallWindowLimits);
    GrowthCapture smallWindowCapture;
    uint64_t smallWindowSubkeys = 0;
    const auto sliced = smallWindowGrowth.runRequestGrowthWindows(deviceProjection,
        deviceTasks, hostTasks, deviceFilter, growthSchedule,
        {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
         analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0},
        captureGrowth, &smallWindowCapture, &smallWindowSubkeys, 1);
    ASSERT_TRUE(sliced.candidateNodeWindowQueries > 0);
    ASSERT_TRUE(sliced.scheduledGateNodes < sliced.unslicedGateNodes);
    ASSERT_EQ(sliced.acceptedEventCount, 7ull);
    ASSERT_EQ(sliced.rawRequestCount, 4ull);
    ASSERT_EQ(smallWindowSubkeys, 5ull);
    std::vector<std::string> slicedEvents, slicedRequests;
    for (const auto& event : smallWindowCapture.events) slicedEvents.push_back(eventText(event));
    for (const auto& request : smallWindowCapture.requests) slicedRequests.push_back(eventText(request.event));
    std::sort(slicedEvents.begin(), slicedEvents.end());
    std::sort(slicedRequests.begin(), slicedRequests.end());
    ASSERT_EQ(slicedEvents, expectedEvents);
    ASSERT_EQ(slicedRequests, expectedRequests);

    // A store that holds every level keeps the whole frontier on the device:
    // no host page moves, same events and requests.
    auto storeLimits = streamLimits;
    storeLimits.frontierStoreRecords = 64;
    gl::gpu::CudaPhase2GrowthBuffer storeGrowth(storeLimits);
    GrowthCapture storeCapture;
    uint64_t storeSubkeys = 0;
    const auto stored = storeGrowth.runRequestGrowthWindows(deviceProjection,
        deviceTasks, hostTasks, deviceFilter, growthSchedule,
        {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
         analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0},
        captureGrowth, &storeCapture, &storeSubkeys, 1);
    ASSERT_EQ(stored.acceptedEventCount, 7ull);
    ASSERT_EQ(stored.rawRequestCount, 4ull);
    ASSERT_EQ(storeSubkeys, 5ull);
    ASSERT_EQ(stored.frontierSpills, 0ull);
    ASSERT_EQ(stored.uploadedFrontierBytes, 0ull);
    ASSERT_EQ(stored.downloadedFrontierBytes, 0ull);
    ASSERT_TRUE(stored.maximumFrontierCount > storeLimits.frontierRecords);
    std::vector<std::string> storedEvents, storedRequests;
    for (const auto& event : storeCapture.events) storedEvents.push_back(eventText(event));
    for (const auto& request : storeCapture.requests) storedRequests.push_back(eventText(request.event));
    std::sort(storedEvents.begin(), storedEvents.end());
    std::sort(storedRequests.begin(), storedRequests.end());
    ASSERT_EQ(storedEvents, expectedEvents);
    ASSERT_EQ(storedRequests, expectedRequests);

    // Global cutoff and duplicate ownership follow original order, not drains.
    for (bool interrupted : {false, true}) {
        gl::gpu::Phase2EarlyExitLedger ledger(hostProjection, hostTasks, growthSchedule);
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        gl::SealedPageSet* outputs[] = {&pages};
        const auto receive = [&](uint32_t request, uint64_t position, const char* name, bool trigger) {
            gl::FiringRecord record;
            record.rplExpr2 = gl::SealedString::copyFrom(pages, name, static_cast<int32_t>(std::strlen(name)));
            gl::gpu::DevicePhase2FiringRecord source{};
            source.requestIndex = request;
            source.growthPosition = position;
            source.flags = trigger ? gl::gpu::deviceFiringDoomTrigger : 0;
            ledger.observeHeaders(&source, 1);
            if (trigger) ASSERT_TRUE(ledger.retainsHeader(source));
            ledger.receive(source, record);
        };
        if (!interrupted) {
            ASSERT_FALSE(ledger.beginDrain(true, 4, 3));
            receive(0, 1, "A", false);
            receive(1, 2, "B0", true);
            receive(1, 2, "B1", false);
            receive(2, 4, "C", false);
        } else {
            const uint32_t physicalOrder[] = {3, 2, 1, 0};
            for (uint32_t at = 0; at < 4; ++at) {
                const uint32_t order = physicalOrder[at];
                ASSERT_TRUE(ledger.beginDrain(at == 3, 1, 1));
                gl::gpu::DeviceAcceptedGrowthEvent event{};
                event.count = 1;
                event.statementIndices[0] = order == 0 ? 0 : (order == 3 ? 2 : 1);
                event.filteredPositionCodes[0] = static_cast<uint16_t>(order + 1);
                event.flags = 7;
                ledger.appendEvents(&event, 1);
                const gl::gpu::DeviceOrderedRequestToken token{1, 0, 0};
                ledger.appendRequests(&token, 1);
                if (order == 0) receive(0, 1, "A", false);
                else if (order == 3) receive(0, 1, "C", false);
                else {
                    receive(0, 1, order == 1 ? "B0" : "late-B0", true);
                    receive(0, 1, order == 1 ? "B1" : "late-B1", false);
                }
            }
        }
        int64_t cutoff = 0;
        ASSERT_EQ(ledger.finish(outputs, 1, &cutoff), 3ull);
        pages.seal();
        ASSERT_EQ(cutoff, static_cast<int64_t>(2ull << 14));
        ASSERT_EQ(ledger.evaluatedRequests, 3ull);
        std::vector<std::string> names;
        pages.forEachRecord<gl::FiringRecord>([&](const auto& record) { names.push_back(record.rplExpr2.toStdString()); });
        std::sort(names.begin(), names.end());
        ASSERT_EQ(names, (std::vector<std::string>{"A", "B0", "B1"}));
        pages.freePages();
    }

    {
        gl::gpu::Phase2EarlyExitLedger empty(hostProjection, hostTasks, growthSchedule);
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        gl::SealedPageSet* outputs[] = {&pages};
        int64_t cutoff = 0;
        ASSERT_EQ(empty.finish(outputs, 1, &cutoff), 0ull);
        ASSERT_EQ(cutoff, gl::kNoDoomLine);
        pages.seal();
        pages.freePages();
    }
    {
        gl::gpu::Phase2TaskProjectionArena cutoffTasks(taskLimits);
        gl::gpu::Phase2GrowthScheduleArena cutoffSchedule(2);
        for (uint32_t part = 0; part < 2; ++part) {
            gl::ExpressionStump stump{};
            stump.allIdx[0] = static_cast<gl::NameId>(part);
            stump.count = 1;
            cutoffTasks.appendTask(0, &batch, 1, &stump, 1, static_cast<gl::NameId>(part), 2,
                analyzer.parameters.maxIterationNumberVariable, 0);
            cutoffSchedule.appendCall(part, cutoffTasks.tasks[part].batchOffset, 0);
        }
        for (bool interrupted : {false, true}) {
            gl::gpu::Phase2EarlyExitLedger ledger(hostProjection, cutoffTasks, cutoffSchedule);
            gl::SealedPageSet pages;
            pages.bind(&gl::staticMemory());
            gl::SealedPageSet* outputs[] = {&pages};
            if (!interrupted) ASSERT_FALSE(ledger.beginDrain(true, 2, 2));
            for (int part = 1; part >= 0; --part) {
                if (interrupted) {
                    ASSERT_TRUE(ledger.beginDrain(part == 0, 1, 1));
                    gl::gpu::DeviceAcceptedGrowthEvent event{};
                    event.callIndex = static_cast<uint32_t>(part);
                    event.count = 1;
                    event.statementIndices[0] = part;
                    event.filteredPositionCodes[0] = 1;
                    event.flags = 7;
                    ledger.appendEvents(&event, 1);
                    const gl::gpu::DeviceOrderedRequestToken token{1, 0, 0};
                    ledger.appendRequests(&token, 1);
                }
                gl::FiringRecord record;
                record.rplExpr2 = gl::SealedString::copyFrom(pages, part ? "loser" : "winner", part ? 5 : 6);
                gl::gpu::DevicePhase2FiringRecord source{};
                source.requestIndex = interrupted ? 0 : static_cast<uint32_t>(part);
                source.partOrdinal = static_cast<uint32_t>(part);
                source.growthPosition = 1;
                source.flags = gl::gpu::deviceFiringDoomTrigger;
                ledger.receive(source, record);
                if (part == 0) { source.flags = 0; ledger.receive(source, record); }
            }
            int64_t cutoff = 0;
            ASSERT_EQ(ledger.finish(outputs, 1, &cutoff), 2ull);
            ASSERT_EQ(cutoff, static_cast<int64_t>(1ull << 14));
            pages.seal();
            pages.forEachRecord<gl::FiringRecord>([&](const auto& record) {
                ASSERT_EQ(record.rplExpr2.toStdString(), std::string("winner"));
            });
            pages.freePages();
        }
    }

    // Transfer pages do not split a fitting frontier execution window.
    for (uint32_t copies : {1031u, gl::gpu::phase2FrontierCopyRows + 7}) {
        const gl::gpu::Phase2FilterScheduleCapacity manyFilterLimits{copies, copies * 64, copies * 64, 64};
        gl::gpu::Phase2FilterScheduleArena manyFilterSchedule(manyFilterLimits);
        gl::gpu::Phase2GrowthScheduleArena manySchedule(copies);
        for (uint32_t call = 0; call < copies; ++call) {
            manyFilterSchedule.appendCall(0, gl::gpu::DeviceHashMemoryKind::overall,
                analyzer.parameters.maxIterationNumberVariable, 1, block.statementCount);
            manySchedule.appendCall(0, 0, call);
        }
        gl::gpu::CudaPhase2FilterSortBuffer manyFilter(manyFilterLimits);
        ASSERT_EQ(manyFilter.filterAndSort(deviceProjection, manyFilterSchedule), 5u);
        gl::gpu::Phase2GrowthCapacity manyLimits{copies, 64, 65536, 65536,
            32768, 2097152, 1048576, 1048576, 65536, 65536};
        std::vector<std::string> expectedMany;
        for (uint32_t call = 0; call < copies; ++call)
            for (const auto& event : expectedEvents)
                expectedMany.push_back(std::to_string(call) + ":" + event);
        std::sort(expectedMany.begin(), expectedMany.end());
        for (bool constrained : {false, true}) {
            if (constrained && copies != 1031) continue;
            if (constrained) manyLimits.prefixPayloadValues = 512;
            gl::gpu::CudaPhase2GrowthBuffer manyGrowth(manyLimits);
            GrowthCapture captured;
            const auto capture = [](void* opaque, const gl::gpu::CudaPhase2GrowthBuffer& buffer, bool) {
                auto& captured = *static_cast<GrowthCapture*>(opaque);
                std::vector<gl::gpu::DeviceAcceptedGrowthEvent> rows(65536);
                const auto count = buffer.downloadAcceptedEvents(rows.data(), 65536);
                captured.events.insert(captured.events.end(), rows.begin(), rows.begin() + count);
            };
            uint64_t subkeys = 0;
            const auto streamedMany = manyGrowth.runRequestGrowthWindows(deviceProjection,
                deviceTasks, hostTasks, manyFilter, manySchedule,
                {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
                 analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0},
                capture, &captured, &subkeys, 1);
            ASSERT_EQ(streamedMany.acceptedEventCount, copies * 7ull);
            ASSERT_EQ(streamedMany.rawRequestCount, copies * 4ull);
            ASSERT_EQ(subkeys, copies * 5ull);
            ASSERT_EQ(streamedMany.drainCount, 1ull);
            ASSERT_TRUE(streamedMany.frontierTransferRows > 0);
            ASSERT_TRUE(streamedMany.frontierTransferRows <= gl::gpu::phase2FrontierCopyRows);
            // Every level fits the store: no frontier row crosses to the host.
            ASSERT_EQ(streamedMany.frontierTransferCalls, 0ull);
            ASSERT_EQ(streamedMany.frontierSpills, 0ull);
            ASSERT_EQ(streamedMany.uploadedFrontierBytes, 0ull);
            ASSERT_EQ(streamedMany.downloadedFrontierBytes, 0ull);
            if (constrained) { ASSERT_TRUE(streamedMany.prefixWindows > 3); }
            else { ASSERT_EQ(streamedMany.prefixWindows, 3ull); }
            std::vector<std::string> actual;
            for (const auto& event : captured.events)
                actual.push_back(std::to_string(event.callIndex) + ":" + eventText(event));
            std::sort(actual.begin(), actual.end());
            ASSERT_EQ(actual, expectedMany);
            ASSERT_TRUE(streamedMany.probeWindows > 0);
            ASSERT_TRUE(streamedMany.exactCountReads > 0);
        }
        if (copies != 1031) {
            // A request ceiling below one candidate window's survivors keeps two
            // half-size probe windows in flight: the bounds tighten from lagged
            // counter snapshots, requests still drain exactly, every event matches.
            auto requestLimits = manyLimits;
            requestLimits.rawRequests = 8192;
            gl::gpu::CudaPhase2GrowthBuffer requestGrowth(requestLimits);
            GrowthCapture requestCapture;
            const auto captureEvents = [](void* opaque, const gl::gpu::CudaPhase2GrowthBuffer& buffer, bool) {
                auto& captured = *static_cast<GrowthCapture*>(opaque);
                std::vector<gl::gpu::DeviceAcceptedGrowthEvent> rows(65536);
                const auto count = buffer.downloadAcceptedEvents(rows.data(), 65536);
                captured.events.insert(captured.events.end(), rows.begin(), rows.begin() + count);
            };
            uint64_t requestSubkeys = 0;
            const auto bounded = requestGrowth.runRequestGrowthWindows(deviceProjection,
                deviceTasks, hostTasks, manyFilter, manySchedule,
                {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
                 analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0},
                captureEvents, &requestCapture, &requestSubkeys, 1);
            ASSERT_EQ(bounded.acceptedEventCount, copies * 7ull);
            ASSERT_EQ(bounded.rawRequestCount, copies * 4ull);
            ASSERT_EQ(requestSubkeys, copies * 5ull);
            ASSERT_TRUE(bounded.drainCount > 1);
            ASSERT_TRUE(bounded.probeWindows > bounded.candidateWindows);
            std::vector<std::string> actual;
            for (const auto& event : requestCapture.events)
                actual.push_back(std::to_string(event.callIndex) + ":" + eventText(event));
            std::sort(actual.begin(), actual.end());
            ASSERT_EQ(actual, expectedMany);
        }
    }

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
    orderingLimits.requests = 64;
    gl::gpu::CudaPhase2OrderingBuffer ordering(orderingLimits);
    const gl::gpu::Phase2OrderingResult ordered =
        ordering.orderRequests(
            deviceProjection, deviceTasks, growth);
    ASSERT_EQ(ordered.orderedEventCount, 7u);
    ASSERT_EQ(ordered.requestCount, 4u);
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
        ordered.requestCount);
    ordering.downloadOrderedRequests(
        orderedRequests.data(),
        static_cast<uint32_t>(orderedRequests.size()));
    gl::gpu::DeviceAcceptedGrowthEvent eventWindow[2];
    growth.downloadAcceptedEventWindow(2, 2, eventWindow);
    ASSERT_EQ(std::memcmp(eventWindow, events.data() + 2, sizeof(eventWindow)), 0);
    growth.downloadAcceptedEventWindow(static_cast<uint32_t>(events.size()), 0, nullptr);
    gl::gpu::DeviceOrderedRequestToken requestWindow[2];
    ordering.downloadOrderedRequestWindow(1, 2, requestWindow);
    ASSERT_EQ(std::memcmp(requestWindow, orderedRequests.data() + 1, sizeof(requestWindow)), 0);
    ordering.downloadOrderedRequestWindow(static_cast<uint32_t>(orderedRequests.size()), 0, nullptr);
    const uint64_t expectedGrowthPositions[4] = { 3, 4, 5, 5 };
    const uint32_t expectedEventOrders[4] = { 2, 3, 5, 6 };
    const std::string expectedOrderedRequests[4] = {
        "1:2:7", "2:1,2:7", "2:0,2:6", "3:0,1,2:6"
    };
    for (uint32_t index = 0; index < ordered.requestCount; ++index) {
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
         statementIndex < ((body).intEncodedStatements).size(); ++statementIndex) {
        const gl::IntEncodedExpr& expression =
            ((body).intEncodedStatements)[statementIndex];
        ((body).intKnownStatements).upsert(
            gl::StatementKey{ expression.originalId, expression.validityId },
            gl::StatementFlags{ true, false });
        if (statementIndex != 2) {
            const int level = statementIndex == 0
                ? 2 : (statementIndex == 1 ? 3 : 7);
            ((body).intStatementLevelsMap).assignSet(
                gl::StatementKey{
                    expression.originalId, expression.validityId },
                &level, 1);
        }
    }
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
    deviceProjection.upload(hostProjection);
    // CPU and CUDA must reject an impure marker even when its owner is an
    // OR branch. A goal-driven marked-head implication (justification
    // `goalDemand`) is exempt on both routes
    // (D-574); classifying the argument as a
    // recursion product makes the ordinary marker legal. Neither changes any
    // ordinary head inference.
    ((((body).overallHashMemory)).remainingArgsReverseIndex).rebuildReverseIndex(
        ((((body).overallHashMemory)).remainingArgsNormalizedEncodedMap), ownerArena);
    for (int purityPass = 0; purityPass < 3; ++purityPass) {
        const bool goalDemand = purityPass == 1;
        const bool recursionProduct = purityPass == 2;
        const bool markerFires = goalDemand || recursionProduct;
        assignEvaluationValues(2, goalDemand
            ? gl::RuleJustification::goalDemand : gl::RuleJustification::implication);
        if (recursionProduct)
            ((((body).overallHashMemory)).productsOfRecursionIds).mint(variableVSecondary);
        hostProjection.clear();
        hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
        deviceProjection.upload(hostProjection);
        ASSERT_EQ(evaluation.expandEvaluationWork(deviceProjection, deviceTasks,
            growth, ordering).localValueCount, 5u);
        gl::gpu::Phase2OutputUsage purityBounds[5]{};
        evaluation.measureFiringOutputBounds(deviceProjection, deviceTasks,
            growth, ordering, 0, 5, purityBounds);
        uint64_t boundedRecords = 0;
        for (const auto& bound : purityBounds)
            boundedRecords += bound.columns[0];
        ASSERT_EQ(boundedRecords, markerFires ? 5ull : 4ull);
        ASSERT_EQ(evaluation.materializeFiringExpressions(deviceProjection,
            deviceTasks, growth, ordering).firingRecordCount,
            markerFires ? 5u : 4u);

        gl::SealedPageSet processorPages;
        processorPages.bind(&gl::staticMemory());
        for (std::size_t index = 0; index < evaluationRequests.size(); ++index) {
            gl::StaticRequest request{};
            request.count = static_cast<int32_t>(evaluationRequests[index].size());
            request.maxIteration = -1;
            for (int32_t premise = 0; premise < request.count; ++premise)
                request.intExprs[premise] = &((body).intEncodedStatements)[
                    evaluationRequests[index][premise]];
            const auto& key = rawEvaluationKeys[index];
            request.normalizedKey = gl::IntNormalizedKey(key.numberExpressions,
                key.data.data(), static_cast<int32_t>(key.data.size()));
            analyzer.ruleEvaluator.checkLocalEncodedMemoryStatic(request, body,
                gl::scratchArenas().slotCount() - 1, processorPages);
        }
        uint32_t processorHeads = 0;
        uint32_t processorMarkers = 0;
        processorPages.seal();
        processorPages.forEachRecord<gl::FiringRecord>([&](const auto& record) {
            if (record.isMarker) {
                ++processorMarkers;
                ASSERT_EQ(record.validityName.toStdString(), names.decode(markerOrBranchId));
            }
            else ++processorHeads;
        });
        ASSERT_EQ(processorHeads, 4u);
        ASSERT_EQ(processorMarkers, markerFires ? 1u : 0u);
        processorPages.freePages();
    }
    // Growth above exercised a real secondary argument. The evaluation fixture
    // now classifies it as a recursion product; a zero-secondary limit keeps
    // W and its shadow excluded during the later streaming-controller twins.
    analyzer.parameters.maxNumberSecondaryVariables = 0;
    analyzer.parameters.maxNumberSecondaryVariablesOrint = 0;
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

    gl::gpu::Phase2OutputUsage measuredOutputs[5]{};
    evaluation.measureFiringOutputBounds(deviceProjection, deviceTasks, growth,
        ordering, 0, 5, measuredOutputs);
    for (uint32_t index = 0; index < 5; ++index) {
        ASSERT_TRUE(measuredOutputs[index].columns[0] <= 1);
        ASSERT_TRUE(measuredOutputs[index].columns[1] > 0);
    }

    // Force both owner and value expansion to cross their processing bounds.
    auto windowLimits = evaluationLimits;
    windowLimits.reverseOwners = 1;
    windowLimits.candidateOwners = 1;
    windowLimits.encodedHits = 1;
    windowLimits.localValues = 1;
    gl::gpu::CudaPhase2EvaluationBuffer windowed(windowLimits);
    windowed.beginEvaluationWindows(deviceProjection, deviceTasks, growth, ordering);
    std::vector<std::pair<uint32_t, uint32_t>> windowValues;
    uint32_t windowFirings = 0;
    while (windowed.nextEvaluationValues(deviceProjection, deviceTasks, growth, ordering)) {
        gl::gpu::DeviceEvaluationValueWork work{};
        ASSERT_EQ(windowed.downloadValueWork(&work, 1), 1u);
        windowValues.emplace_back(work.requestIndex, work.blobRecordIndex);
        const auto produced = windowed.materializeFiringExpressions(
            deviceProjection, deviceTasks, growth, ordering);
        windowFirings += produced.firingRecordCount;
    }
    std::vector<std::pair<uint32_t, uint32_t>> expectedWindowValues;
    for (const auto& work : valueWork)
        expectedWindowValues.emplace_back(work.requestIndex, work.blobRecordIndex);
    std::sort(windowValues.begin(), windowValues.end());
    std::sort(expectedWindowValues.begin(), expectedWindowValues.end());
    ASSERT_EQ(windowValues, expectedWindowValues);
    ASSERT_EQ(windowFirings, 5u);
    // Exhaustion closes the continuation; the same frozen image may be replayed.
    windowed.beginEvaluationWindows(deviceProjection, deviceTasks, growth, ordering);
    uint32_t repeatedValues = 0;
    while (windowed.nextEvaluationValues(deviceProjection, deviceTasks, growth, ordering))
        ++repeatedValues;
    ASSERT_EQ(repeatedValues, 5u);

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
    ASSERT_EQ(sealing.hostBytes(), uint64_t(0));   // construction allocates nothing
    gl::SealedPageSet sealedOutput;
    sealedOutput.bind(&gl::staticMemory());
    gl::SealedPageSet* sealedOutputs[1] = { &sealedOutput };
    ASSERT_EQ(sealing.downloadAndSeal(
        evaluation, *sealingTwin(hostProjection), firingExpressions, 5,
        sealedOutputs, 1), 5u);
    ASSERT_TRUE(sealedOutput.sealed());
    ASSERT_EQ(sealedOutput.recordCount(), 5);
    {
        // The first download owns one measured lease for all output ceilings.
        const uint64_t expectedSealingBytes = gl::gpu::Phase2SealingArena::hostLayout(
            evaluationLimits).roundedBytes(gl::gpuHostStorage().blockBytes());
        ASSERT_EQ(sealing.hostBytes(), expectedSealingBytes);
    }
    const auto sealedContent = [](const gl::FiringRecord& record) {
        std::vector<std::string> values;
        const auto number = [&](auto value) { values.push_back(std::to_string(value)); };
        const auto strings = [&](const auto& run) {
            number(run.size());
            for (const auto& text : run) values.push_back(text.toStdString());
        };
        number(record.isMarker); number(record.isOrdis2Demand);
        number(record.isGoalNearMiss);
        values.push_back(record.rplExpr2.toStdString());
        values.push_back(record.validityName.toStdString());
        number(record.levels.size());
        for (int level : record.levels) number(level);
        values.push_back(record.originTag.toStdString());
        number(record.originDeps.size());
        for (const auto& dependency : record.originDeps) {
            values.push_back(dependency.original.toStdString());
            values.push_back(dependency.validityName.toStdString());
        }
        number(record.doNotDisintegrate); number(record.allowOrDisintegration);
        number(record.recursionHead); number(record.alreadyKnown);
        number(record.iteration); number(record.demandSourceImplId);
        strings(record.markerArgsSorted); strings(record.admv.key);
        strings(record.admv.remainingArgsSorted);
        number(record.admv.standardMaxAdmissionDepth);
        number(record.admv.standardMaxSecondaryNumber);
        number(record.admv.flag); number(record.admv.ordisOnly);
        return values;
    };
    std::vector<std::vector<std::string>> expectedSealedContent;
    std::vector<std::string> expectedSealedMarker;
    uint32_t sealedIndex = 0;
    sealedOutput.forEachRecord<gl::FiringRecord>(
        [&](const gl::FiringRecord& sealed) {
            expectedSealedContent.push_back(sealedContent(sealed));
            if (sealed.isMarker) expectedSealedMarker = sealedContent(sealed);
            ASSERT_TRUE(sealedIndex < firingOrder.size());
            const gl::gpu::DevicePhase2FiringRecord& source =
                firingRecords[firingOrder[sealedIndex]];
            ASSERT_EQ(sealed.rplExpr2.toStdString(), sliceText(source.expression));
            ASSERT_EQ(sealed.validityName.toStdString(),
                ((body).nameMap).decode(source.validityId));
            const bool sourceMarker = (source.flags
                & gl::gpu::deviceFiringMarker) != 0;
            ASSERT_EQ(sealed.isMarker, sourceMarker);
            ASSERT_FALSE(sealed.isOrdis2Demand);
            ASSERT_FALSE(sealed.isGoalNearMiss);
            ASSERT_EQ(sealed.recursionHead, !sourceMarker);
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
                        ? ((body).ruleInterner).decode(
                            static_cast<int32_t>(expected.originalId))
                        : ((body).nameMap).decode(expected.originalId);
                    ASSERT_EQ(sealed.originDeps[
                        static_cast<int32_t>(dependency)].original.toStdString(),
                        expectedOriginal);
                    ASSERT_EQ(sealed.originDeps[
                        static_cast<int32_t>(dependency)].validityName.toStdString(),
                        ((body).nameMap).decode(expected.validityId));
                }
                ASSERT_EQ(sealed.doNotDisintegrate,
                    (source.flags
                        & gl::gpu::deviceFiringDoNotDisintegrate) != 0);
                ASSERT_EQ(sealed.allowOrDisintegration,
                    (source.flags
                        & gl::gpu::deviceFiringAllowOrDisintegration) != 0);

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
                        ((body).ruleInterner).decode(markerRemaining[
                            source.markerRemainingArgOffset + argument]));
                }
                ASSERT_EQ(sealed.admv.standardMaxAdmissionDepth,
                    source.standardMaxAdmissionDepth);
                ASSERT_EQ(sealed.admv.standardMaxSecondaryNumber,
                    source.standardMaxSecondaryNumber);
                ASSERT_FALSE(sealed.admv.flag);
                ASSERT_EQ(sealed.admv.ordisOnly,
                    (source.flags & gl::gpu::deviceFiringOrdisOnly) != 0);

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
                | gl::gpu::deviceFiringOrdisOnly));
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
            | gl::gpu::deviceFiringAlreadyKnown
            | gl::gpu::deviceFiringRecursionHead;
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
                ((body).intEncodedStatements)[
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

    // Bounds cover all output columns for each independently emitted value.
    gl::SealedPageSet streamedOutput;
    streamedOutput.bind(&gl::staticMemory());
    gl::SealedPageSet* streamedOutputs[] = {&streamedOutput};
    for (uint32_t valueIndex = 0; valueIndex < valueWork.size(); ++valueIndex) {
        const auto single = evaluation.materializeFiringExpressions(
            deviceProjection, deviceTasks, growth, ordering, valueIndex, 1);
        const uint64_t actual[] = {single.firingRecordCount, single.generatedByteCount,
            single.levelValueCount, single.originDependencyCount, single.markerKeyCount,
            single.markerRemainingArgCount, single.markerArgCount};
        for (uint32_t column = 0; column < 7; ++column)
            ASSERT_TRUE(actual[column] <= measuredOutputs[valueIndex].columns[column]);
        ASSERT_EQ(sealing.downloadAndSeal(evaluation, *sealingTwin(hostProjection), single,
            single.firingRecordCount, streamedOutputs, 1, {false, false}), single.firingRecordCount);
        ASSERT_TRUE(streamedOutput.filling());
    }
    ASSERT_EQ(streamedOutput.recordCount(), 5);
    streamedOutput.seal();
    std::vector<std::vector<std::string>> actualSealedContent;
    streamedOutput.forEachRecord<gl::FiringRecord>([&](const auto& record) {
        actualSealedContent.push_back(sealedContent(record));
    });
    std::sort(actualSealedContent.begin(), actualSealedContent.end());
    std::sort(expectedSealedContent.begin(), expectedSealedContent.end());
    ASSERT_EQ(actualSealedContent, expectedSealedContent);
    streamedOutput.freePages();
    uint32_t markerValue = static_cast<uint32_t>(valueWork.size());
    std::vector<std::string> expectedMarkerKeys;
    for (const auto& record : firingRecords) {
        if ((record.flags & gl::gpu::deviceFiringMarker) == 0) continue;
        for (uint32_t index = 0; index < valueWork.size(); ++index) {
            if (valueWork[index].requestIndex == record.requestIndex
                && valueWork[index].blobRecordIndex == record.blobRecordIndex)
                markerValue = index;
        }
        for (uint32_t key = 0; key < record.markerKeyCount; ++key)
            expectedMarkerKeys.push_back(sliceText(markerKeys[record.markerKeyOffset + key]));
    }
    ASSERT_TRUE(markerValue < valueWork.size());
    ASSERT_EQ(expectedMarkerKeys.size(), std::size_t{2});
    gl::SealedPageSet fragmentedOutput;
    fragmentedOutput.bind(&gl::staticMemory());
    gl::SealedPageSet* fragmentedOutputs[] = {&fragmentedOutput};
    gl::SealedSpanBuilder<gl::SealedString> markerBuilder(fragmentedOutput, 2);
    struct MarkerAssembly {
        gl::SealedSpanBuilder<gl::SealedString>* keys;
        gl::FiringRecord record{};
        bool started{false};
    } markerAssembly{&markerBuilder};
    gl::gpu::Phase2SealingOptions fragmentOptions{false, false};
    fragmentOptions.context = &markerAssembly;
    fragmentOptions.receive = [](void* context, uint32_t blockIndex, const gl::FiringRecord& record,
        const gl::gpu::DevicePhase2FiringRecord&) {
        auto& assembly = *static_cast<MarkerAssembly*>(context);
        ASSERT_EQ(blockIndex, 0u);
        ASSERT_TRUE(record.isMarker);
        if (assembly.started) {
            ASSERT_EQ(assembly.record.rplExpr2.data(), record.rplExpr2.data());
            ASSERT_EQ(assembly.record.markerArgsSorted.data(), record.markerArgsSorted.data());
            ASSERT_EQ(assembly.record.admv.remainingArgsSorted.data(), record.admv.remainingArgsSorted.data());
        } else {
            assembly.record = record;
            assembly.started = true;
        }
        assembly.keys->append(record.admv.key.data(), record.admv.key.size());
    };
    std::vector<std::string> segmentedMarkerKeys;
    for (uint32_t key = 0; key < expectedMarkerKeys.size(); ++key) {
        gl::gpu::Phase2OutputUsage bound{};
        evaluation.measureFiringOutputBounds(deviceProjection, deviceTasks, growth,
            ordering, markerValue, 1, &bound, key, 1);
        ASSERT_EQ(bound.columns[4], 1ull);
        const auto fragment = evaluation.materializeFiringExpressions(
            deviceProjection, deviceTasks, growth, ordering, markerValue, 1, key, 1);
        ASSERT_EQ(fragment.firingRecordCount, 1u);
        ASSERT_EQ(fragment.markerKeyCount, 1u);
        ASSERT_EQ(fragment.markerRemainingArgCount, 2u);
        ASSERT_EQ(fragment.markerArgCount, 1u);
        ASSERT_EQ(fragment.generatedByteCount, bound.columns[1]);
        fragmentOptions.repeatedMarkerHeader = markerAssembly.started ? &markerAssembly.record : nullptr;
        ASSERT_EQ(sealing.downloadAndSeal(evaluation, *sealingTwin(hostProjection), fragment,
            1, fragmentedOutputs, 1, fragmentOptions), 1u);
        ASSERT_TRUE(fragmentedOutput.filling());
        ASSERT_EQ(fragmentedOutput.recordCount(), 0);
        std::vector<char> bytes(fragment.generatedByteCount);
        ASSERT_EQ(evaluation.downloadGeneratedBytes(bytes.data(), fragment.generatedByteCount),
            fragment.generatedByteCount);
        gl::gpu::DeviceEvaluationByteSlice keySlice{};
        ASSERT_EQ(evaluation.downloadMarkerKeys(&keySlice, 1), 1u);
        segmentedMarkerKeys.emplace_back(bytes.data() + keySlice.offset, keySlice.length);
    }
    ASSERT_EQ(segmentedMarkerKeys, expectedMarkerKeys);
    markerAssembly.record.admv.key = markerBuilder.finish();
    fragmentedOutput.appendRecord(markerAssembly.record);
    fragmentedOutput.seal();
    ASSERT_EQ(fragmentedOutput.recordCount(), 1);
    ASSERT_EQ(sealedContent(markerAssembly.record), expectedSealedMarker);
    fragmentedOutput.freePages();

    auto plannedLimits = evaluationLimits;
    plannedLimits.firingRecords = 1;
    plannedLimits.generatedBytes = 128;
    plannedLimits.levelValues = 4;
    plannedLimits.originDependencies = 4;
    plannedLimits.markerKeys = 1;
    plannedLimits.markerRemainingArgs = 2;
    plannedLimits.markerArgs = 1;
    gl::gpu::CudaPhase2EvaluationBuffer plannedEvaluation(plannedLimits);
    ASSERT_EQ(plannedEvaluation.expandEvaluationWork(deviceProjection, deviceTasks,
        growth, ordering).localValueCount, 5u);
    gl::gpu::Phase2SealingArena plannedSealing(plannedLimits);
    gl::SealedPageSet plannedOutput;
    plannedOutput.bind(&gl::staticMemory());
    gl::SealedPageSet* plannedOutputs[] = {&plannedOutput};
    gl::SealedSpanBuilder<gl::SealedString> plannedMarkerKeys(plannedOutput, 2);
    MarkerAssembly plannedMarker{&plannedMarkerKeys};
    gl::gpu::Phase2FiringCursor firingCursor{};
    uint32_t plannedDownloads = 0;
    uint32_t plannedFragments = 0;
    for (;;) {
        const auto window = plannedEvaluation.nextFiringWindow(
            deviceProjection, deviceTasks, growth, ordering, firingCursor);
        if (window.state == gl::gpu::Phase2FiringWindowState::complete) break;
        ASSERT_EQ(window.state, gl::gpu::Phase2FiringWindowState::ready);
        ASSERT_TRUE(plannedOutput.filling());
        if (window.markerFragment) {
            ASSERT_EQ(window.totalMarkerKeys, 2u);
            ASSERT_EQ(window.markerKeyBegin, plannedFragments);
            ASSERT_EQ(window.markerKeyCount, 1u);
            auto options = fragmentOptions;
            options.context = &plannedMarker;
            options.repeatedMarkerHeader = plannedMarker.started ? &plannedMarker.record : nullptr;
            ASSERT_EQ(plannedSealing.downloadAndSeal(plannedEvaluation, *sealingTwin(hostProjection),
                window.used, window.used.firingRecordCount, plannedOutputs, 1, options), 1u);
            ++plannedFragments;
            if (firingCursor.markerKey == 0) {
                plannedMarker.record.admv.key = plannedMarkerKeys.finish();
                plannedOutput.appendRecord(plannedMarker.record);
            }
        } else {
            plannedSealing.downloadAndSeal(plannedEvaluation, *sealingTwin(hostProjection),
                window.used, window.used.firingRecordCount, plannedOutputs, 1, {false, false});
        }
        ++plannedDownloads;
    }
    ASSERT_EQ(plannedDownloads, 6u);
    ASSERT_EQ(plannedFragments, 2u);
    ASSERT_EQ(firingCursor.value, 5u);
    ASSERT_EQ(firingCursor.markerKey, 0u);
    ASSERT_EQ(plannedOutput.recordCount(), 5);
    plannedOutput.seal();
    actualSealedContent.clear();
    plannedOutput.forEachRecord<gl::FiringRecord>([&](const auto& record) {
        actualSealedContent.push_back(sealedContent(record));
    });
    std::sort(actualSealedContent.begin(), actualSealedContent.end());
    ASSERT_EQ(actualSealedContent, expectedSealedContent);
    plannedOutput.freePages();
    ASSERT_EQ(plannedEvaluation.nextFiringWindow(deviceProjection, deviceTasks,
        growth, ordering, firingCursor).state, gl::gpu::Phase2FiringWindowState::complete);

    // Exercise the production controller with every request in a separate drain.
    gl::gpu::CudaPhase2GrowthBuffer controllerGrowth(streamLimits);
    gl::gpu::CudaPhase2OrderingBuffer controllerOrdering({2, 1});
    auto controllerLimits = plannedLimits;
    controllerLimits.requests = 1;
    controllerLimits.reverseOwners = controllerLimits.candidateOwners = 1;
    controllerLimits.encodedHits = controllerLimits.localValues = 1;
    gl::gpu::CudaPhase2EvaluationBuffer controllerEvaluation(controllerLimits);
    gl::gpu::Phase2SealingArena controllerSealing(controllerLimits);
    gl::gpu::Phase2BufferPlan controllerPlan{};
    controllerPlan.growth = streamLimits;
    controllerPlan.evaluation = controllerLimits;
    gl::gpu::Phase2StreamBuffers controllerBuffers{deviceProjection, deviceTasks,
        deviceFilter, controllerGrowth, controllerOrdering, controllerEvaluation};
    gl::SealedPageSet controllerOutput;
    controllerOutput.bind(&gl::staticMemory());
    gl::SealedPageSet* controllerOutputs[] = {&controllerOutput};
    uint64_t controllerSubkeys = 0;
    const auto controlled = controllerSealing.runWindows(controllerBuffers,
        *sealingTwin(hostProjection), hostTasks, growthSchedule,
        {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
         analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0}, controllerPlan,
        controllerOutputs, 1, &controllerSubkeys, 1, nullptr, diskStorage);
    ASSERT_EQ(controlled.evaluatedRequests, 4ull);
    ASSERT_EQ(controlled.firingRecords, 5ull);
    ASSERT_EQ(controllerSubkeys, 5ull);
    ASSERT_TRUE(controlled.growth.drainCount > 1);
    ASSERT_TRUE(controlled.outputDownloads >= 6);
    ASSERT_TRUE(controlled.outputTransferBytes > 0);
    actualSealedContent.clear();
    controllerOutput.forEachRecord<gl::FiringRecord>([&](const auto& record) {
        actualSealedContent.push_back(sealedContent(record));
    });
    std::sort(actualSealedContent.begin(), actualSealedContent.end());
    ASSERT_EQ(actualSealedContent, expectedSealedContent);
    controllerOutput.freePages();

    // Enabling level stops without a discharge preserves the complete stream.
    {
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        gl::SealedPageSet* outputs[] = {&pages};
        uint64_t subkeys = 0;
        int64_t stop = 0;
        const auto full = controllerSealing.runWindows(controllerBuffers,
            *sealingTwin(hostProjection), hostTasks, growthSchedule,
            {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
             analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0},
            controllerPlan, outputs, 1, &subkeys, 1, &stop);
        ASSERT_EQ(stop, INT64_MAX);
        ASSERT_EQ(full.firingRecords, controlled.firingRecords);
        ASSERT_EQ(subkeys, controllerSubkeys);
        std::vector<std::vector<std::string>> content;
        pages.forEachRecord<gl::FiringRecord>([&](const auto& record) {
            content.push_back(sealedContent(record));
        });
        std::sort(content.begin(), content.end());
        ASSERT_EQ(content, expectedSealedContent);
        pages.freePages();
    }

    // Insufficient minimum fragment capacity is a scheduling state, not a write.
    for (uint32_t column : {1u, 2u, 3u, 5u}) {
        auto blockedLimits = evaluationLimits;
        if (column == 1) blockedLimits.generatedBytes = 1;
        if (column == 2) blockedLimits.levelValues = 1;
        if (column == 3) blockedLimits.originDependencies = 1;
        if (column == 5) blockedLimits.markerRemainingArgs = 1;
        gl::gpu::CudaPhase2EvaluationBuffer blocked(blockedLimits);
        ASSERT_EQ(blocked.expandEvaluationWork(deviceProjection, deviceTasks,
            growth, ordering).localValueCount, 5u);
        uint32_t blockedValue = 0;
        while (measuredOutputs[blockedValue].columns[column] <= 1) {
            ++blockedValue;
            ASSERT_TRUE(blockedValue < 5);
        }
        gl::gpu::Phase2FiringCursor blockedCursor{blockedValue, 0};
        for (uint32_t repeat = 0; repeat < 2; ++repeat) {
            const auto window = blocked.nextFiringWindow(deviceProjection, deviceTasks,
                growth, ordering, blockedCursor);
            ASSERT_EQ(window.state, gl::gpu::Phase2FiringWindowState::needsCapacity);
            ASSERT_TRUE(window.required.columns[column] > 1);
            ASSERT_EQ(blockedCursor.value, blockedValue);
            ASSERT_EQ(blockedCursor.markerKey, 0u);
            ASSERT_EQ(window.used.firingRecordCount, 0u);
        }
    }
    ASSERT_EQ(evaluation.materializeFiringExpressions(
        deviceProjection, deviceTasks, growth, ordering).firingRecordCount, 5u);

    // Metadata transfer pages must not become execution or download boundaries.
    {
        constexpr uint32_t plainValues = 2053;
        gl::LocalMemoryValue plain;
        plain.valueId = gpuHeadId;
        plain.originalImplicationId = gpuSourceId;
        plain.justification = gl::RuleJustification::implication;
        plain.validityId = mainId;
        plain.levels = {4, 5};
        ((((body).overallHashMemory)).encodedMap).assignRun(
            mappedEvaluationKey(evaluationRequests[0]),
            std::vector<gl::LocalMemoryValue>(plainValues, plain));
        const auto bulkInput = gl::gpu::makePhase2BufferPlanInput(
            gl::gpu::measurePhase2ProjectionUsage(body, analyzer.anchorInfo, 0x0fu, true), {}, 0, 0, 0);
        gl::gpu::Phase2ProjectionArena bulkProjection(bulkInput.projection);
        bulkProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
        gl::gpu::CudaPhase2ProjectionBuffer bulkDevice(bulkInput.projection);
        bulkDevice.upload(bulkProjection);
        auto bulkLimits = evaluationLimits;
        bulkLimits.localValues = bulkLimits.firingRecords = 4096;
        bulkLimits.generatedBytes = 1u << 20;
        bulkLimits.levelValues = bulkLimits.originDependencies = 1u << 15;
        gl::gpu::CudaPhase2EvaluationBuffer bulkEvaluation(bulkLimits);
        const auto expanded = bulkEvaluation.expandEvaluationWork(bulkDevice, deviceTasks, growth, ordering);
        ASSERT_EQ(expanded.localValueCount, plainValues + 4);
        const auto complete = bulkEvaluation.materializeFiringExpressions(
            bulkDevice, deviceTasks, growth, ordering);
        ASSERT_EQ(complete.firingRecordCount, plainValues + 4);
        ASSERT_EQ(bulkEvaluation.orderFiringRecords(bulkDevice), plainValues + 4);
        gl::gpu::Phase2SealingArena bulkSealing(bulkLimits);
        gl::SealedPageSet referencePages;
        referencePages.bind(&gl::staticMemory());
        gl::SealedPageSet* referenceOutputs[] = {&referencePages};
        bulkSealing.downloadAndSeal(bulkEvaluation, *sealingTwin(bulkProjection), complete,
            complete.firingRecordCount, referenceOutputs, 1);
        std::vector<std::vector<std::string>> referenceContent;
        referencePages.forEachRecord<gl::FiringRecord>([&](const auto& record) {
            referenceContent.push_back(sealedContent(record));
        });
        std::sort(referenceContent.begin(), referenceContent.end());
        referencePages.freePages();
        gl::gpu::Phase2FiringCursor bulkCursor{};
        const auto bulkWindow = bulkEvaluation.nextFiringWindow(
            bulkDevice, deviceTasks, growth, ordering, bulkCursor);
        ASSERT_EQ(bulkWindow.state, gl::gpu::Phase2FiringWindowState::ready);
        ASSERT_EQ(bulkWindow.valueCount, plainValues + 4);
        ASSERT_EQ(bulkWindow.used.firingRecordCount, complete.firingRecordCount);
        ASSERT_TRUE(bulkWindow.valueCount > 2 * gl::gpu::CudaPhase2EvaluationBuffer::outputPlanningRows);
        ASSERT_EQ(bulkEvaluation.nextFiringWindow(bulkDevice, deviceTasks, growth, ordering,
            bulkCursor).state, gl::gpu::Phase2FiringWindowState::complete);
        gl::SealedPageSet bulkPages;
        bulkPages.bind(&gl::staticMemory());
        gl::SealedPageSet* bulkOutputs[] = {&bulkPages};
        bulkSealing.downloadAndSeal(bulkEvaluation, *sealingTwin(bulkProjection), bulkWindow.used,
            bulkWindow.used.firingRecordCount, bulkOutputs, 1, {false, true});
        std::vector<std::vector<std::string>> bulkContent;
        bulkPages.forEachRecord<gl::FiringRecord>([&](const auto& record) {
            bulkContent.push_back(sealedContent(record));
        });
        std::sort(bulkContent.begin(), bulkContent.end());
        ASSERT_EQ(bulkContent, referenceContent);
        bulkPages.freePages();

        // An actual output limit still drains at the exact fitting boundary.
        auto cutLimits = bulkLimits;
        cutLimits.firingRecords = 2048;
        gl::gpu::CudaPhase2EvaluationBuffer cutEvaluation(cutLimits);
        cutEvaluation.expandEvaluationWork(bulkDevice, deviceTasks, growth, ordering);
        gl::gpu::Phase2FiringCursor cutCursor{};
        const auto firstCut = cutEvaluation.nextFiringWindow(bulkDevice, deviceTasks, growth, ordering, cutCursor);
        ASSERT_EQ(firstCut.valueCount, 2048u);
        ASSERT_EQ(firstCut.used.firingRecordCount, 2048u);
        ASSERT_EQ(cutCursor.value, 2048u);
        gl::gpu::Phase2SealingArena cutSealing(cutLimits);
        gl::SealedPageSet cutPages;
        cutPages.bind(&gl::staticMemory());
        gl::SealedPageSet* cutOutputs[] = {&cutPages};
        cutSealing.downloadAndSeal(cutEvaluation, *sealingTwin(bulkProjection), firstCut.used,
            firstCut.used.firingRecordCount, cutOutputs, 1, {false, false});
        const auto lastCut = cutEvaluation.nextFiringWindow(bulkDevice, deviceTasks, growth, ordering, cutCursor);
        ASSERT_EQ(lastCut.valueCount, plainValues + 4 - 2048);
        ASSERT_EQ(cutCursor.value, plainValues + 4);
        cutSealing.downloadAndSeal(cutEvaluation, *sealingTwin(bulkProjection), lastCut.used,
            lastCut.used.firingRecordCount, cutOutputs, 1, {false, true});
        bulkContent.clear();
        cutPages.forEachRecord<gl::FiringRecord>([&](const auto& record) {
            bulkContent.push_back(sealedContent(record));
        });
        std::sort(bulkContent.begin(), bulkContent.end());
        ASSERT_EQ(bulkContent, referenceContent);
        cutPages.freePages();
        ((((body).overallHashMemory)).encodedMap).assignRun(
            mappedEvaluationKey(evaluationRequests[0]), std::vector<gl::LocalMemoryValue>{plain});
    }

    ((body).intToBeProved).assignSet(
        gl::packStatementKey(knownSecondaryHead, mainId), nullptr, 0);
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
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
    evaluation.markDoomTriggers(deviceProjection);
    std::vector<gl::gpu::DevicePhase2FiringRecord> triggerHeaders(5);
    ASSERT_EQ(evaluation.downloadFiringRecords(triggerHeaders.data(), 5), 5u);
    uint32_t triggers = 0;
    for (const auto& header : triggerHeaders) {
        if ((header.flags & gl::gpu::deviceFiringDoomTrigger) != 0) {
            ASSERT_EQ(header.validityId, mainId);
            ASSERT_EQ(header.requestIndex, 0u);
            ++triggers;
        }
    }
    ASSERT_EQ(triggers, 1u);
    // Observing a discharge does not remove other outputs or repeated firings
    // from the current level. Only a committed stop excludes a later download.
    {
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        gl::SealedPageSet* outputs[] = {&pages};
        int64_t stop = INT64_MAX;
        gl::gpu::Phase2SealingOptions options{false, false};
        options.blockStops = &stop;
        for (uint32_t copy = 0; copy < 2; ++copy) {
            ASSERT_EQ(sealing.downloadAndSeal(evaluation, *sealingTwin(hostProjection),
                firingExpressions, 5, outputs, 1, options), 5u);
            ASSERT_EQ(stop, gl::gpu::phase2PendingBlockStop);
        }
        ASSERT_EQ(pages.recordCount(), 10);
        stop = 0;
        ASSERT_EQ(sealing.downloadAndSeal(evaluation, *sealingTwin(hostProjection),
            firingExpressions, 5, outputs, 1, options), 0u);
        ASSERT_EQ(pages.recordCount(), 10);
        pages.seal();
        std::vector<std::vector<std::string>> content;
        pages.forEachRecord<gl::FiringRecord>([&](const auto& record) {
            content.push_back(sealedContent(record));
        });
        std::sort(content.begin(), content.end());
        auto repeated = expectedSealedContent;
        repeated.insert(repeated.end(), expectedSealedContent.begin(), expectedSealedContent.end());
        std::sort(repeated.begin(), repeated.end());
        ASSERT_EQ(content, repeated);
        pages.freePages();
    }
    {
        gl::SealedPageSet earlyOutput;
        earlyOutput.bind(&gl::staticMemory());
        gl::SealedPageSet* earlyOutputs[] = {&earlyOutput};
        uint64_t subkeys = 0;
        int64_t cutoff = 0;
        const auto controlledEarly = controllerSealing.runWindows(controllerBuffers,
            *sealingTwin(hostProjection), hostTasks, growthSchedule,
            {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
             analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0},
            controllerPlan, earlyOutputs, 1, &subkeys, 1, &cutoff, diskStorage);
        ASSERT_TRUE(controlledEarly.growth.drainCount > 1);
        ASSERT_TRUE(controlledEarly.growth.prunedFrontierNodes > 0);
        ASSERT_TRUE(controlledEarly.growth.rawRequestCount < controlled.growth.rawRequestCount);
        ASSERT_EQ(controlledEarly.firingRecords, static_cast<uint64_t>(goalRetainedCount));
        ASSERT_EQ(cutoff, 0ll);
        ASSERT_EQ(earlyOutput.recordCount(), 1);
        earlyOutput.forEachRecord<gl::FiringRecord>([&](const auto& record) {
            ASSERT_EQ(record.rplExpr2.toStdString(), std::string("(gpuHead[secondary_v_shadow,static])"));
        });
        earlyOutput.freePages();
    }

    uint32_t goalDoomRequest = 5;
    ASSERT_EQ(evaluation.downloadDoomRequestIndices(
        &goalDoomRequest, 1), 1u);
    ASSERT_EQ(goalDoomRequest, 0u);
    uint32_t retainedGoalRecord = 5;
    ASSERT_EQ(evaluation.downloadFiringOrder(&retainedGoalRecord, 1), 1u);
    ASSERT_TRUE(retainedGoalRecord < firingRecords.size());
    ASSERT_EQ(firingRecords[retainedGoalRecord].requestIndex, 0u);
    ASSERT_EQ(firingRecords[retainedGoalRecord].growthPosition, 3ull);

    ((body).intToBeProved).resetToFresh();
    ((body).intToBeProved).assignSet(
        gl::packStatementKey(knownProductHead, mainId), nullptr, 0);
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
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

    // A two-premise discharge keeps every result through that level, including
    // all marker fragments, independently of request and output capacities.
    std::vector<std::vector<std::string>> levelContent;
    for (bool small : {true, false}) {
        auto limits = streamLimits;
        if (!small) {
            limits.acceptedEvents = 16;
            limits.rawRequests = 8;
        }
        gl::gpu::CudaPhase2GrowthBuffer levelGrowth(limits);
        gl::gpu::CudaPhase2OrderingBuffer levelOrdering({limits.acceptedEvents, limits.rawRequests});
        const auto outputLimits = small ? controllerLimits : evaluationLimits;
        gl::gpu::CudaPhase2EvaluationBuffer levelEvaluation(outputLimits);
        gl::gpu::Phase2SealingArena levelSealing(outputLimits);
        gl::gpu::Phase2StreamBuffers buffers{deviceProjection, deviceTasks,
            deviceFilter, levelGrowth, levelOrdering, levelEvaluation};
        auto plan = controllerPlan;
        plan.growth = limits;
        plan.evaluation = outputLimits;
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        gl::SealedPageSet* outputs[] = {&pages};
        uint64_t subkeys = 0;
        int64_t stop = INT64_MAX;
        const auto result = levelSealing.runWindows(buffers, *sealingTwin(hostProjection),
            hostTasks, growthSchedule,
            {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
             analyzer.parameters.maxNumberSecondaryVariablesOrint, 4, 0},
            plan, outputs, 1, &subkeys, 1, &stop,
            small ? diskStorage : gl::gpu::RequestFrontierStorageOptions{});
        ASSERT_EQ(stop, 0ll);
        ASSERT_EQ(result.firingRecords, 4ull);
        ASSERT_EQ(result.evaluatedRequests, 3ull);
        ASSERT_TRUE(result.growth.prunedFrontierNodes > 0);
        std::vector<std::vector<std::string>> content;
        uint32_t markers = 0;
        pages.forEachRecord<gl::FiringRecord>([&](const auto& record) {
            content.push_back(sealedContent(record));
            if (record.isMarker) {
                ++markers;
                ASSERT_EQ(sealedContent(record), expectedSealedMarker);
            }
        });
        ASSERT_EQ(markers, 1u);
        std::sort(content.begin(), content.end());
        if (small) levelContent = content;
        else ASSERT_EQ(content, levelContent);
        pages.freePages();
    }

    analyzer.parameters.compressor_mode = true;
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
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
    ((body).intToBeProved).resetToFresh();
    ((body).primedForContradiction) = true;
    ((body).intKnownStatements).upsert(
        gl::StatementKey{ negatedSecondaryHead, mainId },
        gl::StatementFlags{ true, false });
    hostProjection.clear();
    hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters, 0x0fu, true);
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
    capturedGrowth.events.clear();
    capturedGrowth.requests.clear();
    const auto streamedStumps = streamingGrowth.runRequestGrowthWindows(deviceProjection,
        deviceTasks, hostTasks, deviceFilter, growthSchedule,
        {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
         analyzer.parameters.maxNumberSecondaryVariablesOrint, 1, 0},
        captureGrowth, &capturedGrowth, &streamedSubkeyCount, 1, nullptr, diskStorage);
    ASSERT_EQ(streamedStumps.acceptedEventCount, 3ull);
    ASSERT_EQ(streamedStumps.rawRequestCount, 2ull);
    ASSERT_EQ(streamedSubkeyCount, 2ull);
    streamedEvents.clear();
    for (const auto& event : capturedGrowth.events) streamedEvents.push_back(eventText(event));
    std::sort(streamedEvents.begin(), streamedEvents.end());
    ASSERT_EQ(streamedEvents, expectedStumpEvents);
    // Roomy windows seed every stump of both calls in one pair launch; the
    // second call on the same task repeats each record under its own index.
    {
        gl::gpu::Phase2FilterScheduleArena twoFilterSchedule(filterLimits);
        for (int call = 0; call < 2; ++call)
            twoFilterSchedule.appendCall(0, gl::gpu::DeviceHashMemoryKind::overall,
                analyzer.parameters.maxIterationNumberVariable, 1, block.statementCount);
        gl::gpu::CudaPhase2FilterSortBuffer twoFilter(filterLimits);
        ASSERT_EQ(twoFilter.filterAndSort(deviceProjection, twoFilterSchedule), 5u);
        gl::gpu::Phase2GrowthScheduleArena twoCalls(4);
        twoCalls.appendCall(0, 0, 0);
        twoCalls.appendCall(0, 0, 1);
        gl::gpu::CudaPhase2GrowthBuffer roomyGrowth(growthLimits);
        const auto captureRoomy = [](void* context, const gl::gpu::CudaPhase2GrowthBuffer& buffer, bool) {
            auto& captured = *static_cast<GrowthCapture*>(context);
            std::vector<gl::gpu::DeviceAcceptedGrowthEvent> eventRows(128);
            std::vector<gl::gpu::DeviceRawGrowthRequest> requestRows(64);
            const uint32_t eventCount = buffer.downloadAcceptedEvents(eventRows.data(), 128);
            const uint32_t requestCount = buffer.downloadRawRequests(requestRows.data(), 64);
            captured.events.insert(captured.events.end(), eventRows.begin(), eventRows.begin() + eventCount);
            captured.requests.insert(captured.requests.end(), requestRows.begin(), requestRows.begin() + requestCount);
        };
        GrowthCapture roomyCapture;
        uint64_t roomySubkeys = 0;
        const auto roomy = roomyGrowth.runRequestGrowthWindows(deviceProjection,
            deviceTasks, hostTasks, twoFilter, twoCalls,
            {analyzer.parameters.maxLenHypoKey, analyzer.parameters.maxNumberSecondaryVariables,
             analyzer.parameters.maxNumberSecondaryVariablesOrint, 1, 0},
            captureRoomy, &roomyCapture, &roomySubkeys, 1);
        ASSERT_EQ(roomyCapture.requests.size(), 4u);
        ASSERT_EQ(roomy.acceptedEventCount, 6ull);
        ASSERT_EQ(roomy.rawRequestCount, 4ull);
        ASSERT_EQ(roomySubkeys, 4ull);
        std::vector<std::string> roomyEvents, expectedRoomyEvents;
        for (const auto& event : roomyCapture.events)
            roomyEvents.push_back(std::to_string(event.callIndex) + ":" + eventText(event));
        for (const char* call : {"0:", "1:"})
            for (const auto& event : expectedStumpEvents) expectedRoomyEvents.push_back(call + event);
        std::sort(roomyEvents.begin(), roomyEvents.end());
        std::sort(expectedRoomyEvents.begin(), expectedRoomyEvents.end());
        ASSERT_EQ(roomyEvents, expectedRoomyEvents);
    }
    const gl::gpu::Phase2OrderingResult orderedStumps =
        ordering.orderRequests(
            deviceProjection, deviceTasks, growth);
    ASSERT_EQ(orderedStumps.orderedEventCount, 3u);
    ASSERT_EQ(orderedStumps.requestCount, 2u);
    std::vector<uint32_t> orderedStumpIndices(
        orderedStumps.orderedEventCount);
    ordering.downloadOrderedEventIndices(
        orderedStumpIndices.data(),
        static_cast<uint32_t>(orderedStumpIndices.size()));
    ASSERT_EQ(eventText(stumpEvents[orderedStumpIndices[0]]), "1:2:7");
    ASSERT_EQ(eventText(stumpEvents[orderedStumpIndices[1]]), "2:0,1:1");
    ASSERT_EQ(eventText(stumpEvents[orderedStumpIndices[2]]), "3:0,1,2:6");
    std::vector<gl::gpu::DeviceOrderedRequestToken> orderedStumpRequests(
        orderedStumps.requestCount);
    ordering.downloadOrderedRequests(
        orderedStumpRequests.data(),
        static_cast<uint32_t>(orderedStumpRequests.size()));
    ASSERT_EQ(orderedStumpRequests[0].growthPosition, 1ull);
    ASSERT_EQ(orderedStumpRequests[0].eventOrder, 0u);
    ASSERT_EQ(orderedStumpRequests[1].growthPosition, 2ull);
    ASSERT_EQ(orderedStumpRequests[1].eventOrder, 2u);
}

CUDA_TEST(phase2_projection, cuda_ordering_keeps_equal_requests_within_and_across_drains) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.maxIterationNumberVariable = 3;
    gl::Memory body;
    body.setExprKey("cuda_ordering_repeats");
    gl::NameMap& names = ((body).nameMap);
    const gl::NameId mainId = names.encode("main");
    const gl::NameId nameId = names.encode("duplicate_name");
    const gl::NameId originalId = names.encode("(duplicate_name[x])");
    const gl::NameId argumentId = names.encode("duplicate_argument");
    const gl::IntEncodedExpr statement = projectionStatement(
        nameId, originalId, mainId, argumentId);
    ((body).intEncodedStatements).push_back(statement);
    ((body).intEncodedStatements).push_back(statement);
    gl::tests::admitEveryStatementShape(body);
    const gl::IntEncodedExpr* expressions[1] = {
        &((body).intEncodedStatements)[0]
    };
    gl::NameId keyData[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
    const gl::NameId keyLength = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(
        expressions, 1, keyData, gl::ExecutionParameters::MAX_KEY_SLOTS);
    const gl::NormKey key{
        1, std::vector<gl::NameId>(keyData, keyData + keyLength) };
    gl::OwnerSet owner;
    owner.hasLooseOwner = true;
    const int32_t fixtureRuleId = ((body).ruleInterner).encode(
        "(cuda-ordering-rule[1])");
    const gl::RuleOwner fixtureOwner = gl::packRuleOwner(
        fixtureRuleId, mainId);
    owner.owners.emplace_back(fixtureOwner, -1);
    gl::ScratchArena& ownerArena = gl::genScratchArenas().forSlot(
        gl::genScratchArenas().slotCount() - 1);
    ((((body).overallHashMemory)).normalizedEncodedSubkeys).assignRun(
        key, std::vector<gl::OwnerSet>{ owner });
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(
        ((((body).overallHashMemory)).normalizedEncodedKeys),
        key.numberExpressions, key.data.data(),
        static_cast<int32_t>(key.data.size()), fixtureOwner, ownerArena);
    ((((body).overallHashMemory)).maxKeyLength) = 1;

    uint32_t nameBytes = 0;
    for (gl::NameId id = 1; id <= names.nameCount(); ++id)
        nameBytes += static_cast<uint32_t>(names.decodeView(id).len);
    const gl::gpu::Phase2ProjectionCapacity projectionLimits =
        projectionCapacity(1, 2,
            static_cast<uint32_t>(names.nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena hostProjection(projectionLimits);
    const gl::gpu::DeviceLogicalBlockProjection block =
        hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
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
    growthLimits.frontierStoreRecords = 16;
    gl::gpu::CudaPhase2GrowthBuffer growth(growthLimits);
    const gl::gpu::Phase2GrowthResult growthResult = growth.runRequestGrowth(
        deviceProjection, deviceTasks, deviceFilter, growthSchedule,
        gl::gpu::DevicePhase2GrowthParameters{
            8, 8, 8, gl::gpu::kDeviceGrowthProductionCooperativeSpan, 0 });
    ASSERT_EQ(growthResult.acceptedEventCount, 4u);
    ASSERT_EQ(growthResult.rawRequestCount, 4u);

    gl::gpu::Phase2OrderingCapacity orderingLimits{};
    orderingLimits.events = 16;
    orderingLimits.requests = 8;
    gl::gpu::CudaPhase2OrderingBuffer ordering(orderingLimits);
    const gl::gpu::Phase2OrderingResult result =
        ordering.orderRequests(
            deviceProjection, deviceTasks, growth);
    ASSERT_EQ(result.orderedEventCount, 4u);
    ASSERT_EQ(result.requestCount, 4u);
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
    std::vector<gl::gpu::DeviceOrderedRequestToken> requests(4);
    ASSERT_EQ(ordering.downloadOrderedRequests(requests.data(), 4), 4u);
    for (uint32_t i = 0; i < 4; ++i) {
        ASSERT_EQ(requests[i].eventOrder, i);
        ASSERT_EQ(requests[i].growthPosition, static_cast<uint64_t>(i + 1));
        ASSERT_EQ(events[requests[i].eventIndex].callIndex, i / 2);
        ASSERT_EQ(events[requests[i].eventIndex].statementIndices[0], static_cast<int32_t>(i % 2));
    }
    // Repeated invocations do not remember or suppress earlier request keys.
    ASSERT_EQ(ordering.orderRequests(deviceProjection, deviceTasks, growth).requestCount, 4u);
    ASSERT_EQ(ordering.downloadOrderedRequests(requests.data(), 4), 4u);

    // Moving physical task/batch slots preserves all four requests.
    hostTasks.appendTask(0, batches, 2, nullptr, 0, 0, 0,
        analyzer.parameters.maxIterationNumberVariable, 0);
    deviceTasks.upload(hostTasks);
    for (auto& call : growthSchedule.calls) {
        call.taskIndex = 1;
        call.batchIndex += 2;
    }
    growth.runRequestGrowth(deviceProjection, deviceTasks, deviceFilter, growthSchedule,
        {8, 8, 8, gl::gpu::kDeviceGrowthProductionCooperativeSpan, 0});
    ASSERT_EQ(ordering.orderRequests(deviceProjection, deviceTasks, growth).requestCount, 4u);
    hostTasks.tasks[1].stumpOrdinal = 1;
    hostTasks.tasks[1].stumpTotal = 2;
    deviceTasks.upload(hostTasks);
    ASSERT_EQ(ordering.orderRequests(deviceProjection, deviceTasks, growth).requestCount, 4u);

    auto tinyLimits = growthLimits;
    tinyLimits.frontierRecords = 1;
    tinyLimits.frontierStoreRecords = 1;
    tinyLimits.acceptedEvents = 1;
    tinyLimits.rawRequests = 1;
    tinyLimits.candidateWindowRecords = 1;
    gl::gpu::CudaPhase2GrowthBuffer tinyGrowth(tinyLimits);
    struct RequestDrain {
        const gl::gpu::CudaPhase2ProjectionBuffer* projection;
        const gl::gpu::CudaPhase2TaskBuffer* tasks;
        gl::gpu::CudaPhase2OrderingBuffer* ordering;
        uint32_t retained{0};
    } drainContext{&deviceProjection, &deviceTasks, &ordering};
    const auto drainRequests = [](void* context, const gl::gpu::CudaPhase2GrowthBuffer& stream, bool) {
        auto& state = *static_cast<RequestDrain*>(context);
        state.retained += state.ordering->orderRequests(
            *state.projection, *state.tasks, stream).requestCount;
    };
    uint64_t streamedCounts[2]{};
    const auto streamed = tinyGrowth.runRequestGrowthWindows(deviceProjection, deviceTasks,
        hostTasks, deviceFilter, growthSchedule,
        {8, 8, 8, gl::gpu::kDeviceGrowthProductionCooperativeSpan, 0},
        drainRequests, &drainContext, streamedCounts, 2);
    ASSERT_EQ(streamed.acceptedEventCount, 4ull);
    ASSERT_EQ(streamed.rawRequestCount, 4ull);
    ASSERT_EQ(streamed.drainCount, 4ull);
    ASSERT_EQ(drainContext.retained, 4u);
    ASSERT_EQ(streamedCounts[0], 0ull);
    ASSERT_EQ(streamedCounts[1], 4ull);
}

CUDA_TEST(phase2_projection, repeated_large_request_windows_need_no_host_history) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory body;
    body.setExprKey("history_pages");
    const auto mainId = ((body).nameMap).encode("main");
    const auto nameId = ((body).nameMap).encode("historyPredicate");
    for (uint32_t i = 0; i < 129; ++i) {
        const auto variable = "v" + std::to_string(i);
        const auto original = "(historyPredicate[" + variable + "])";
        ((body).intEncodedStatements).push_back(projectionStatement(nameId,
            ((body).nameMap).encode(original), mainId, ((body).nameMap).encode(variable)));
    }
    gl::tests::admitEveryStatementShape(body);
    const auto keyFor = [&](int count) {
        const gl::IntEncodedExpr* premises[] = {
            &((body).intEncodedStatements)[0], &((body).intEncodedStatements)[1]};
        gl::NameId data[gl::ExecutionParameters::MAX_KEY_SLOTS]{};
        const auto size = gl::HashKeyBuilder::makeIntNormalizedKeyFromEncoded(
            premises, count, data, gl::ExecutionParameters::MAX_KEY_SLOTS);
        return gl::NormKey{count, std::vector<gl::NameId>(data, data + size)};
    };
    const auto rule = gl::packRuleOwner(((body).ruleInterner).encode("(historyRule[1,2])"), mainId);
    gl::OwnerSet owner;
    owner.hasLooseOwner = true;
    owner.owners.emplace_back(rule, -1);
    ((((body).overallHashMemory)).normalizedEncodedSubkeys).assignRun(keyFor(1), std::vector<gl::OwnerSet>{owner});
    const auto pairKey = keyFor(2);
    auto& scratch = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::tests::HashRuleInstallerFixture::addWholeKeyOwner(((((body).overallHashMemory)).normalizedEncodedKeys),
        pairKey.numberExpressions, pairKey.data.data(), static_cast<int32_t>(pairKey.data.size()), rule, scratch);
    ((((body).overallHashMemory)).maxKeyLength) = 2;
    uint32_t nameBytes = 0;
    for (gl::NameId i = 1; i <= ((body).nameMap).nameCount(); ++i)
        nameBytes += static_cast<uint32_t>(((body).nameMap).decodeView(i).len);
    const auto projectionLimits = projectionCapacity(1, 129,
        static_cast<uint32_t>(((body).nameMap).nameCount() + 1), nameBytes);
    gl::gpu::Phase2ProjectionArena hostProjection(projectionLimits);
    const auto block = hostProjection.appendLogicalBlock(body, analyzer.definitions, analyzer.anchorInfo, analyzer.parameters);
    gl::gpu::CudaPhase2ProjectionBuffer projection(projectionLimits);
    projection.upload(hostProjection);
    const gl::gpu::Phase2TaskProjectionCapacity taskLimits{1, 1, 1, 1};
    gl::gpu::Phase2TaskProjectionArena hostTasks(taskLimits);
    gl::gpu::Phase2RequestBatchInput batch{};
    batch.memory = gl::gpu::DeviceHashMemoryKind::overall;
    hostTasks.appendTask(0, &batch, 1, nullptr, 0, 0, 0,
        analyzer.parameters.maxIterationNumberVariable, 1);
    gl::gpu::CudaPhase2TaskBuffer tasks(taskLimits);
    tasks.upload(hostTasks);
    const gl::gpu::Phase2FilterScheduleCapacity filterLimits{1, 256, 256, 256};
    gl::gpu::Phase2FilterScheduleArena filterSchedule(filterLimits);
    filterSchedule.appendCall(0, gl::gpu::DeviceHashMemoryKind::overall,
        analyzer.parameters.maxIterationNumberVariable, 1, block.statementCount);
    gl::gpu::CudaPhase2FilterSortBuffer filter(filterLimits);
    ASSERT_EQ(filter.filterAndSort(projection, filterSchedule), 129u);
    gl::gpu::Phase2GrowthScheduleArena schedule(1);
    schedule.appendCall(0, 0, 0);
    gl::gpu::CudaPhase2GrowthBuffer growth({1, 256, 512, 16384, 16384, 8192, 4096, 4096, 16384, 512});
    const auto grown = growth.runRequestGrowth(projection, tasks, filter, schedule, {8, 8, 8, 4, 0});
    constexpr uint32_t pairCount = 129 * 128 / 2;
    ASSERT_EQ(grown.rawRequestCount, pairCount);
    gl::gpu::CudaPhase2OrderingBuffer ordering({16384, 16384});
    ASSERT_EQ(ordering.orderRequests(projection, tasks, growth).requestCount, pairCount);
    const auto hostBlocksBeforeRepeat = gl::staticMemory().blocksInUse();
    ASSERT_EQ(ordering.orderRequests(projection, tasks, growth).requestCount, pairCount);
    ASSERT_EQ(gl::staticMemory().blocksInUse(), hostBlocksBeforeRepeat);

}

CUDA_TEST(phase2_projection, fixed_cuda_phase2_buffers_report_owned_bytes) {
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
    ASSERT_EQ(projection.fixedAllocationBytes(), 270595088ull);   // 262144 statement rows of 240 bytes

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
    capacity.frontierStoreRecords = 1048576;
    gl::gpu::CudaPhase2GrowthBuffer buffer(capacity);
    ASSERT_TRUE(buffer.fixedAllocationBytes() > 800097352ull);
    ASSERT_TRUE(buffer.fixedAllocationBytes() < 1024ull * 1024ull * 1024ull);
    ASSERT_EQ(buffer.fixedAllocationBytes(), 824501863ull);   // includes 2048 per-call keep-flag bytes

    gl::gpu::Phase2OrderingCapacity orderingCapacity{};
    orderingCapacity.events = 2097152;
    orderingCapacity.requests = 131072;
    gl::gpu::CudaPhase2OrderingBuffer ordering(orderingCapacity);
    ASSERT_GE(ordering.fixedAllocationBytes(), 79691780ull);
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
    if (evaluationBytes != 38899251ull) {
        std::fprintf(stderr, "[phase2-gpu] evaluator bytes=%llu\n",
            static_cast<unsigned long long>(evaluationBytes));
    }
    ASSERT_EQ(evaluationBytes, 38899251ull);
    ASSERT_TRUE(evaluation.fixedAllocationBytes()
        < 64ull * 1024ull * 1024ull);
}

// The level-end cutoff removes exactly the nodes of cut-off calls and keeps the
// survivors in order, across frontier pages, as the node-by-node walk did.
CUDA_TEST(phase2_projection, prune_frontier_by_call_matches_the_node_walk) {
    gl::LbArena arena(&gl::staticMemory());
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::gpu::DeviceGrowthNode> frontier(&arena, &dirty);
    std::mt19937 random(20260925u);
    constexpr uint32_t callCount = 7;
    for (int trial = 0; trial < 8; ++trial) {
        frontier.clear();
        std::vector<gl::gpu::DeviceGrowthNode> nodes;
        const int32_t size = trial == 0 ? 0 : 1 + static_cast<int32_t>(random() % 900);
        for (int32_t i = 0; i < size; ++i) {
            gl::gpu::DeviceGrowthNode node{};
            node.callIndex = static_cast<uint32_t>(random() % callCount);
            node.runOrdinal = static_cast<uint32_t>(i);
            node.count = 1 + static_cast<gl::NameId>(random() % 4);
            node.startPosition = static_cast<gl::NameId>(3 * i);
            node.filteredPositions[0] = static_cast<gl::NameId>(i);
            nodes.push_back(node);
            frontier.push_back(node);
        }
        uint8_t keep[callCount];
        for (uint32_t call = 0; call < callCount; ++call)
            keep[call] = trial == 1 ? 1 : trial == 2 ? 0
                : static_cast<uint8_t>(random() % 2);
        std::vector<gl::gpu::DeviceGrowthNode> expected;
        for (const auto& node : nodes)
            if (keep[node.callIndex] != 0) expected.push_back(node);
        const uint64_t removed = gl::gpu::pruneFrontierByCall(frontier, keep, callCount);
        ASSERT_EQ(removed, static_cast<uint64_t>(nodes.size() - expected.size()));
        ASSERT_EQ(frontier.size(), static_cast<int32_t>(expected.size()));
        for (int32_t i = 0; i < frontier.size(); ++i)
            ASSERT_EQ(std::memcmp(&frontier[i], &expected[static_cast<std::size_t>(i)],
                sizeof(gl::gpu::DeviceGrowthNode)), 0);
    }
    frontier.clear();
}

// The device half of the level-end cutoff keeps exactly the host walk's
// survivors in order, across thread blocks; zero rows are a no-op.
CUDA_TEST(phase2_projection, device_frontier_prune_matches_the_host_prune) {
    std::mt19937 random(20261001u);
    constexpr uint32_t callCount = 9;
    gl::LbArena arena(&gl::staticMemory());
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::gpu::DeviceGrowthNode> frontier(&arena, &dirty);
    for (int trial = 0; trial < 6; ++trial) {
        const uint32_t size = trial == 0 ? 0u : 1u + static_cast<uint32_t>(random() % 5000);
        std::vector<gl::gpu::DeviceGrowthNode> rows(size);
        frontier.clear();
        for (uint32_t i = 0; i < size; ++i) {
            rows[i].callIndex = static_cast<uint32_t>(random() % callCount);
            rows[i].runOrdinal = i;
            rows[i].count = 1 + static_cast<gl::NameId>(random() % 4);
            rows[i].startPosition = static_cast<gl::NameId>(2 * i);
            rows[i].filteredPositions[0] = static_cast<gl::NameId>(i);
            frontier.push_back(rows[i]);
        }
        uint8_t keep[callCount];
        for (uint32_t call = 0; call < callCount; ++call)
            keep[call] = trial == 1 ? 1 : trial == 2 ? 0 : static_cast<uint8_t>(random() % 2);
        std::vector<gl::gpu::DeviceGrowthNode> survivors(size + 1);
        const uint32_t kept = gl::gpu::pruneDeviceFrontierByCallOnHost(rows.data(), size, keep,
            callCount, survivors.data());
        const uint64_t removed = gl::gpu::pruneFrontierByCall(frontier, keep, callCount);
        ASSERT_EQ(static_cast<uint64_t>(kept) + removed, static_cast<uint64_t>(size));
        ASSERT_EQ(static_cast<int32_t>(kept), frontier.size());
        for (uint32_t i = 0; i < kept; ++i)
            ASSERT_EQ(std::memcmp(&survivors[i], &frontier[static_cast<int32_t>(i)],
                sizeof(gl::gpu::DeviceGrowthNode)), 0);
    }
    frontier.clear();
}

// The device packs the three prefix sizes the host schedules windows from; the
// host unpacks exactly the values of the full records, across thread blocks.
CUDA_TEST(phase2_projection, packed_prefix_sizes_round_trip_the_measured_fields) {
    std::mt19937 random(7u);
    std::vector<gl::gpu::DeviceGrowthPrefix> prefixes(1500);
    for (auto& prefix : prefixes) {
        prefix.payloadOffset = static_cast<uint32_t>(random());
        prefix.variableOffset = static_cast<uint32_t>(random());
        prefix.secondaryOffset = static_cast<uint32_t>(random());
        prefix.payloadLength = static_cast<uint16_t>(random());
        prefix.variableCount = static_cast<uint16_t>(random());
        prefix.secondaryCount = static_cast<uint16_t>(random());
        prefix.reserved = static_cast<uint16_t>(random());
        prefix.packedSummary = static_cast<uint32_t>(random());
        prefix.candidateCount = static_cast<uint32_t>(random());
    }
    prefixes[0].payloadLength = prefixes[0].variableCount = prefixes[0].secondaryCount = 0xffffu;
    prefixes[1].payloadLength = prefixes[1].variableCount = prefixes[1].secondaryCount = 0;
    std::vector<uint64_t> packed(prefixes.size() + 1, ~0ull);
    gl::gpu::packPhase2PrefixSizesOnHost(prefixes.data(),
        static_cast<uint32_t>(prefixes.size()), packed.data());
    for (std::size_t i = 0; i < prefixes.size(); ++i) {
        const gl::gpu::Phase2PrefixSizes sizes = gl::gpu::unpackPhase2PrefixSizes(packed[i]);
        ASSERT_EQ(sizes.payloadLength, prefixes[i].payloadLength);
        ASSERT_EQ(sizes.variableCount, prefixes[i].variableCount);
        ASSERT_EQ(sizes.secondaryCount, prefixes[i].secondaryCount);
    }
    ASSERT_EQ(packed[prefixes.size()], ~0ull);   // nothing written past the run
    gl::gpu::packPhase2PrefixSizesOnHost(prefixes.data(), 0, packed.data());
    ASSERT_EQ(packed[prefixes.size()], ~0ull);
}
#endif

TEST(phase2_projection, construction_waves_bound_even_single_worker_ownership) {
    auto capacity = gl::gpu::kFtaShortcutPhase2ProjectionCapacity;
    std::array<gl::gpu::Phase2ProjectionUsage,512> usage{};
    std::array<uint32_t,512> ends{};
    for (auto& row : usage) {
        row.logicalBlocks = 1;
        row.metadataBytes = 48;
    }
    const auto count = gl::gpu::planProjectionConstructionWaves(
        usage.data(),static_cast<uint32_t>(usage.size()),capacity,
        ends.data(),static_cast<uint32_t>(ends.size()));
    ASSERT_EQ(count,2u);
    ASSERT_EQ(ends[0],341u);
    ASSERT_EQ(ends[1],512u);
    uint32_t begin=0;
    for (uint32_t wave=0;wave<count;++wave) {
        uint64_t bytes=0;
        for (uint32_t i=begin;i<ends[wave];++i) bytes+=usage[i].metadataBytes;
        ASSERT_TRUE(bytes<=capacity.metadataBytes);
        begin=ends[wave];
    }
    ASSERT_EQ(begin,512u);
    capacity.statements=3;
    for (uint32_t i=0;i<4;++i) usage[i].statements=2;
    ASSERT_EQ(gl::gpu::planProjectionConstructionWaves(usage.data(),4,
        capacity,ends.data(),static_cast<uint32_t>(ends.size())),4u);
    for (uint32_t i=0;i<4;++i) ASSERT_EQ(ends[i],i+1);
    ASSERT_EQ(gl::gpu::planProjectionConstructionWaves(nullptr,0,capacity,nullptr,0),0u);
}

TEST(projection_column, owning_column_appends_resizes_iterates_and_compares) {
    gl::gpu::ProjectionColumn<int32_t> column;
    ASSERT_EQ(column.capacity(), std::size_t(0));
    ASSERT_EQ(column.ownedBytes(), uint64_t(0));
    column.reserve(8);
    ASSERT_EQ(column.capacity(), std::size_t(8));
    ASSERT_EQ(column.ownedBytes(), uint64_t(8 * sizeof(int32_t)));
    ASSERT_TRUE(column.empty());
    column.push_back(4);
    const int32_t more[3] = { 5, 6, 7 };
    column.insert(column.end(), more, more + 3);
    column.resize(6, -1);
    ASSERT_EQ(column.size(), std::size_t(6));
    ASSERT_EQ(column[0], 4);
    ASSERT_EQ(column[3], 7);
    ASSERT_EQ(column.back(), -1);
    column.resize(7);                    // value-initialized
    ASSERT_EQ(column[6], 0);
    column.resize(5);                    // shrinking lowers the length only
    ASSERT_EQ(column.size(), std::size_t(5));
    column.resize(7);
    ASSERT_EQ(column[5], 0);
    int32_t sum = 0;
    for (const int32_t value : column) sum += value;
    ASSERT_EQ(sum, 4 + 5 + 6 + 7 - 1);
    gl::gpu::ProjectionColumn<int32_t> same;
    same.reserve(7);
    const int32_t values[7] = { 4, 5, 6, 7, -1, 0, 0 };
    same.insert(same.end(), values, values + 7);
    ASSERT_TRUE(column == same);
    same[2] = 9;
    ASSERT_TRUE(!(column == same));
    const gl::gpu::ProjectionColumn<int32_t>& view = column;
    ASSERT_TRUE(view.data() == column.data() && view.end() - view.begin() == 7);
    column.clear();
    ASSERT_TRUE(column.empty());
    ASSERT_EQ(column.capacity(), std::size_t(8));
}

TEST(projection_column, slices_write_into_the_owner_and_move_back_in_place) {
    gl::gpu::ProjectionColumn<char> storage;
    storage.reserve(10);
    gl::gpu::ProjectionColumn<char> first;
    gl::gpu::ProjectionColumn<char> second;
    first.attachSlice(storage, 0, 4);
    second.attachSlice(storage, 4, 6);
    ASSERT_EQ(first.ownedBytes(), uint64_t(0));
    ASSERT_EQ(second.capacity(), std::size_t(6));
    ASSERT_TRUE(second.data() == storage.data() + 4);
    first.push_back('a');
    first.push_back('b');
    const char tail[3] = { 'x', 'y', 'z' };
    second.insert(second.end(), tail, tail + 3);
    ASSERT_EQ(storage.size(), std::size_t(0));   // slices do not change the owner
    storage.appendMoved(first);                  // already in place
    storage.appendMoved(second);                 // closes the two-element gap
    ASSERT_EQ(storage.size(), std::size_t(5));
    ASSERT_EQ(std::string(storage.data(), storage.size()), std::string("abxyz"));
    ASSERT_TRUE(first.empty() && second.empty());
    ASSERT_EQ(storage.ownedBytes(), uint64_t(10));
}

TEST(phase2_projection, projection_capacities_add_field_by_field) {
    static_assert(sizeof(gl::gpu::Phase2ProjectionCapacity) == 24 * sizeof(uint32_t));
    const gl::gpu::Phase2ProjectionCapacity left = projectionCapacity(2, 3, 4, 5);
    gl::gpu::Phase2ProjectionCapacity right = projectionCapacity(1, 1, 1, 1);
    right.metadataBytes = 7;
    const gl::gpu::Phase2ProjectionCapacity sum = gl::gpu::addProjectionCapacity(left, right);
    uint32_t l[24], r[24], t[24];
    std::memcpy(l, &left, sizeof l);
    std::memcpy(r, &right, sizeof r);
    std::memcpy(t, &sum, sizeof t);
    for (int field = 0; field < 24; ++field) ASSERT_EQ(t[field], l[field] + r[field]);
    ASSERT_EQ(sum.logicalBlocks, 3u);
    ASSERT_EQ(sum.metadataBytes, left.metadataBytes + 7u);
    const gl::gpu::Phase2ProjectionCapacity same =
        gl::gpu::addProjectionCapacity(left, gl::gpu::Phase2ProjectionCapacity{});
    ASSERT_EQ(std::memcmp(&same, &left, sizeof left), 0);
}


TEST(phase2_projection, streamed_projection_assertion_probe) {
    std::string mode;
#ifdef _WIN32
    char* value = nullptr;
    size_t length = 0;
    ASSERT_EQ(_dupenv_s(&value, &length, "GL_TEST_INVALID_STREAMED_PROJECTION"), 0);
    if (value != nullptr) mode = value;
    std::free(value);
#else
    if (const char* value = std::getenv("GL_TEST_INVALID_STREAMED_PROJECTION")) mode = value;
#endif
    if (mode.empty()) return;
    const auto capacity = projectionCapacity(1, 1, 2, 8);
    gl::gpu::Phase2ProjectionArena source(capacity);
    gl::gpu::Phase2ProjectionArena::ShardBases bases{};
    if (mode == "offset") {
        gl::gpu::DeviceNameRecord record{};
        record.byteOffset = 1;
        source.nameRecords.push_back(record);
        bases.nameByte = UINT32_MAX;
        source.rebaseForStream(bases);
    } else if (mode == "retained_base") {
        gl::gpu::Phase2ProjectionArena retained(capacity,
            gl::gpu::Phase2ProjectionArena::SealingOnly{});
        bases.block = 1;
        retained.appendSealingProjection(source, bases);
    } else if (mode == "append_before_begin") {
#ifdef GL_CUDA
        gl::gpu::CudaPhase2ProjectionBuffer device(capacity);
        device.appendStreamedUpload(source, bases);
#else
        ASSERT_TRUE(false);
#endif
    } else if (mode == "adopt_stream_offset") {
        // The second shard is placed at block 1; a stream base at the top of
        // the 32-bit range cannot address it.
        const auto two = projectionCapacity(2, 1, 2, 8);
        gl::gpu::Phase2ProjectionArena backing(two, gl::gpu::addProjectionCapacity(two, two));
        gl::gpu::Phase2ProjectionArena firstShard(two, backing, gl::gpu::Phase2ProjectionCapacity{});
        gl::gpu::Phase2ProjectionArena secondShard(two, backing, two);
        firstShard.logicalBlocks.push_back(gl::gpu::DeviceLogicalBlockProjection{});
        gl::gpu::Phase2ProjectionArena* shards[] = { &firstShard, &secondShard };
        uint32_t firstBlocks[2]{};
        bases.block = UINT32_MAX;
        gl::gpu::Phase2ProjectionWorkers workers(1);
        backing.adoptShards(shards, 2, firstBlocks, bases, workers);
    } else if (mode == "workers_nested_job") {
        gl::gpu::Phase2ProjectionWorkers workers(1);
        auto inner = [](uint32_t) {};
        auto outer = [&workers, &inner](uint32_t) { workers.run(1, inner); };
        workers.run(1, outer);
    } else if (mode == "uploader_submit_after_finish") {
#ifdef GL_CUDA
        gl::gpu::CudaPhase2ProjectionBuffer device(capacity);
        gl::gpu::Phase2ProjectionArena retained(capacity,
            gl::gpu::Phase2ProjectionArena::SealingOnly{});
        device.beginStreamedUpload();
        gl::gpu::Phase2WindowUploader uploader(device, retained);
        uploader.finish();
        std::optional<gl::gpu::Phase2ProjectionArena> window;
        window.emplace(capacity);
        uploader.submit(window, bases);
#else
        ASSERT_TRUE(false);
#endif
    } else if (mode == "uploader_wait_after_finish") {
#ifdef GL_CUDA
        gl::gpu::CudaPhase2ProjectionBuffer device(capacity);
        gl::gpu::Phase2ProjectionArena retained(capacity,
            gl::gpu::Phase2ProjectionArena::SealingOnly{});
        device.beginStreamedUpload();
        gl::gpu::Phase2WindowUploader uploader(device, retained);
        uploader.finish();
        uploader.waitUntilUploaded();
#else
        ASSERT_TRUE(false);
#endif
    } else if (mode == "host_plan_overflow") {
#ifdef GL_CUDA
        gl::gpu::Phase2HostFootprint footprint;
        footprint.fixedBytes = UINT64_MAX;
        footprint.executionBytes = 1;
        (void)gl::gpu::selectPhase2HostWindows(footprint, UINT64_MAX);
#else
        ASSERT_TRUE(false);
#endif
    } else if (mode == "host_page_size") {
#ifdef GL_CUDA
        (void)gl::gpu::phase2HostColumnPages(1, 2049);
#else
        ASSERT_TRUE(false);
#endif
    } else if (mode == "uploader_destroyed_unfinished") {
#ifdef GL_CUDA
        gl::gpu::CudaPhase2ProjectionBuffer device(capacity);
        gl::gpu::Phase2ProjectionArena retained(capacity,
            gl::gpu::Phase2ProjectionArena::SealingOnly{});
        device.beginStreamedUpload();
        {
            gl::gpu::Phase2WindowUploader uploader(device, retained);
        }
#else
        ASSERT_TRUE(false);
#endif
    } else {
        ASSERT_TRUE(false);
    }
    ASSERT_TRUE(false);
}

TEST(phase2_projection, pooled_column_keeps_capacity_and_borrowed_slice_ownership) {
    gl::GlobalMemoryManager manager;
    manager.init({8 * 262144LL, 262144, 2048, gl::PoolKind::GpuHost});
    gl::GpuHostArena arena;
    arena.reserve(128, manager);
    gl::gpu::ProjectionColumn<uint32_t> column;
    column.bindStorage(arena);
    column.reserve(16);
    column.assign(3, 7);
    column.pop_back();
    column.push_back(9);
    ASSERT_EQ(column.size(), std::size_t(3));
    ASSERT_EQ(column[0], 7u);
    ASSERT_EQ(column.back(), 9u);
    gl::gpu::ProjectionColumn<uint32_t> slice;
    slice.attachSlice(column, 8, 4);
    slice.push_back(42);
    ASSERT_EQ(slice.ownedBytes(), uint64_t(0));
    ASSERT_EQ(column.ownedBytes(), uint64_t(64));
    column.appendMoved(slice);
    ASSERT_EQ(column.back(), 42u);
    ASSERT_TRUE(slice.empty());
    ASSERT_EQ(column.capacity(), std::size_t(16));
}

TEST(phase2_projection, schedule_layouts_equal_their_physical_pool_leases) {
    gl::gpu::Phase2TaskProjectionCapacity taskCapacity{2, 4, 8, 4};
    gl::gpu::Phase2TaskProjectionArena tasks(taskCapacity);
    ASSERT_EQ(tasks.hostBytes(), gl::gpu::Phase2TaskProjectionArena::hostLayout(
        taskCapacity).roundedBytes(gl::gpuHostStorage().blockBytes()));
    gl::gpu::Phase2FilterScheduleCapacity filterCapacity{4, 16, 8, 4};
    gl::gpu::Phase2FilterScheduleArena filter(filterCapacity);
    ASSERT_EQ(filter.hostBytes(), gl::gpu::Phase2FilterScheduleArena::hostLayout(
        filterCapacity).roundedBytes(gl::gpuHostStorage().blockBytes()));
    gl::gpu::Phase2GrowthScheduleArena growth(4);
    ASSERT_EQ(growth.hostBytes(), gl::gpu::Phase2GrowthScheduleArena::hostLayout(
        4).roundedBytes(gl::gpuHostStorage().blockBytes()));
    ASSERT_EQ(filter.classSlots.size(), std::size_t(8));
    for (std::size_t i = 0; i < filter.classSlots.size(); ++i) ASSERT_EQ(filter.classSlots[i], -1);
}

#ifdef GL_CUDA
TEST(phase2_projection, host_window_selection_obeys_lifetime_peaks) {
    gl::gpu::Phase2HostFootprint footprint;
    footprint.fixedBytes = 10;
    footprint.constructionOneBytes = 30;
    footprint.constructionTwoBytes = 50;
    footprint.executionBytes = 20;
    ASSERT_EQ(gl::gpu::selectPhase2HostWindows(footprint, 60).windows, 2u);
    ASSERT_EQ(gl::gpu::selectPhase2HostWindows(footprint, 59).windows, 1u);
    ASSERT_EQ(gl::gpu::selectPhase2HostWindows(footprint, 40).windows, 1u);
    ASSERT_EQ(gl::gpu::selectPhase2HostWindows(footprint, 39).windows, 0u);
    footprint.executionBytes = 70;
    ASSERT_EQ(gl::gpu::selectPhase2HostWindows(footprint, 79).windows, 0u);
    const auto fit = gl::gpu::selectPhase2HostWindows(footprint, 80);
    ASSERT_EQ(fit.windows, 2u);
    ASSERT_EQ(fit.oneWindowPeakBytes, 80ull);
    ASSERT_EQ(fit.twoWindowPeakBytes, 80ull);
}

TEST(phase2_projection, paged_host_bounds_cover_directory_promotions_and_hash_growth) {
    using namespace gl;
    ASSERT_EQ(gpu::phase2HostColumnPages(0, 4), 0ull);
    ASSERT_TRUE(gpu::phase2HostColumnPages(513 * 512, 4) > 513);
    ASSERT_EQ(gpu::phase2HostArenaBytes(0), 0ull);
    ASSERT_TRUE(gpu::phase2HostArenaBytes(129, 5) >= 7 * 262144ull);
    LbArena shared(&gpuHostStorage()), prefix(&gpuHostStorage());
    DirtyState dirty = DirtyState::Clean;
    ColdHashMap<BytesKeyStore, uint32_t> keys(&shared, &dirty);
    ColdHashSet<BytesKeyStore> prefixes(&prefix, &dirty);
    ColdHashMap<PodKeyStore<uint64_t>, int32_t> transitions(&prefix, &dirty);
    PagedVector<int32_t> heads(&prefix, &dirty);
    struct Transition { int32_t next, predicate, mask; int16_t bound[ExecutionParameters::MAX_ARITY]; };
    PagedVector<Transition> rows(&prefix, &dirty);
    constexpr uint32_t count = 10000;
    for (uint32_t i = 0; i < count; ++i) {
        uint32_t bytes[4]{i, i + 1, i + 2, i + 3};
        const StrSpan key(reinterpret_cast<const char*>(bytes), sizeof(bytes));
        keys.insert(key, i);
        prefixes.mint(key);
        transitions.insert(i, i);
        heads.push_back(i);
        rows.push_back(Transition{});
    }
    gpu::Phase2ProjectionUsage usage;
    usage.byteMapEntries = count;
    usage.byteKeyBytes = count * 16;
    const auto bound = gpu::phase2HostDictionaryBytes(usage);
    ASSERT_TRUE(bound >= static_cast<uint64_t>(shared.blocksHeld() + prefix.blocksHeld()) * 262144);
    ASSERT_TRUE(bound < 16 * 1024 * 1024ull);
    gpu::Phase2GrowthCapacity growth{};
    growth.calls = 3;
    growth.frontierRecords = 65536;
    ASSERT_TRUE(gpu::phase2HostGrowthBytes(growth, true) > gpu::phase2HostGrowthBytes(growth, false));
}

CUDA_TEST(phase2_projection, host_chunk_plan_reduces_overlap_then_complete_families) {
    using namespace gl::gpu;
    Phase2ProjectionUsage families[3]{};
    Phase2ProjectionUsage total{};
    for (auto& family : families) {
        family.logicalBlocks = 1;
        family.nameRecords = 1;
        family.nameBytes = 80 * 1024 * 1024;
        addPhase2HostUsage(total, family);
    }
    const auto input = makePhase2BufferPlanInput(total, {3, 3, 3, 3}, 3, 3, 1);
    const auto device = planPhase2Buffers(input, 1024, initialPhase2WorkDemand(), {}, true);
    ASSERT_EQ(device.state, Phase2BufferPlanState::fits);
    const auto full = planPhase2HostStorage(input, device, {families, 3}, 3, 4ull << 30, true);
    ASSERT_EQ(full.windows, 2u);
    ASSERT_TRUE(full.twoWindowPeakBytes > full.oneWindowPeakBytes);
    const auto serial = planPhase2HostStorage(input, device, {families, 3}, 3,
        full.twoWindowPeakBytes - 1, true);
    ASSERT_EQ(serial.windows, 1u);
    const auto rejected = planPhase2HostStorage(input, device, {families, 3}, 3,
        full.oneWindowPeakBytes - 1, true);
    ASSERT_EQ(rejected.windows, 0u);
    const auto singleInput = makePhase2BufferPlanInput(families[0], {1, 1, 1, 1}, 1, 1, 1);
    const auto singleDevice = planPhase2Buffers(singleInput, 1024,
        initialPhase2WorkDemand(), {}, true);
    const auto smaller = planPhase2HostStorage(singleInput, singleDevice, {families, 1}, 3,
        full.oneWindowPeakBytes - 1, true);
    ASSERT_TRUE(smaller.windows != 0);
    ASSERT_EQ(planPhase2HostStorage(singleInput, singleDevice, {families, 1}, 3, 1, true).windows, 0u);
    const auto directory = phase2HostDirectoryLayout(3, 3, 3);
    ASSERT_TRUE(directory.bytes > 3 * sizeof(DeviceLogicalBlockProjection));
    ASSERT_TRUE(directory.roundedBytes(262144) <= full.fixedBytes);

    families[1].nameBytes = 96 * 1024 * 1024;
    families[2].nameBytes = 112 * 1024 * 1024;
    total = {};
    for (const auto& family : families) addPhase2HostUsage(total, family);
    const auto unevenInput = makePhase2BufferPlanInput(total, {3, 3, 3, 3}, 3, 3, 1);
    const auto unevenDevice = planPhase2Buffers(unevenInput, 1024,
        initialPhase2WorkDemand(), {}, true);
    const auto uneven = planPhase2HostStorage(unevenInput, unevenDevice, {families, 3},
        3, 4ull << 30, true);
    // The third range cannot reuse the first 80 MiB hole while the 96 MiB
    // upload remains live. Its first-fit address span includes that gap.
    ASSERT_TRUE(uneven.constructionTwoBytes > 288 * 1024 * 1024ull);
    ASSERT_TRUE(uneven.constructionOneBytes < 128 * 1024 * 1024ull);
    ASSERT_EQ(planPhase2HostStorage(unevenInput, unevenDevice, {families, 3}, 3,
        uneven.fixedBytes + 250 * 1024 * 1024ull, true).windows, 1u);
}
#endif

// The device twin of `Memory::isInductionHypothesisRule`: a projected block's
// first induction implication and its multiplication catcher both mark a
// firing as the hypothesis head; no id names a rule in other blocks
// (D-566).
TEST(phase2_projection, both_strong_hypotheses_mark_device_heads) {
    gl::gpu::DeviceLogicalBlockProjection block{};
    ASSERT_FALSE(gl::gpu::isInductionHypothesisSource(block, 0));
    ASSERT_FALSE(gl::gpu::isInductionHypothesisSource(block, 7));
    block.recursionAuxyImplId = 7;
    block.recursionProductImplId = 9;
    ASSERT_TRUE(gl::gpu::isInductionHypothesisSource(block, 7));
    ASSERT_TRUE(gl::gpu::isInductionHypothesisSource(block, 9));
    ASSERT_FALSE(gl::gpu::isInductionHypothesisSource(block, 8));
    ASSERT_FALSE(gl::gpu::isInductionHypothesisSource(block, 0));
}
