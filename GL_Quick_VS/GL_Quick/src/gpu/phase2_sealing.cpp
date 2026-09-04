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

// CUDA-route code: the sealing arena downloads the device evaluation buffer,
// whose download methods live in the `.cu` translation unit. A build without
// CUDA support (Makefile default, USE_CUDA=0) has no such buffer and no
// CUDA route, so this unit compiles to nothing there.
#ifdef GL_CUDA

#include "phase2_sealing.hpp"

#include <algorithm>
#include <cassert>
#include <limits>

namespace gl::gpu {

    /// @brief Allocate all processor download and sealing scratch once.
    ///
    /// @details
    /// Mirrors every device output capacity with a fixed-size processor vector
    /// and reserves reusable per-record sealing scratch at the largest relevant
    /// run ceiling. These are the only heap allocations performed by this owner.
    ///
    /// @param fixedCapacity Immutable evaluator element and byte ceilings.
    /// @return An empty reusable sealing arena.
    /// @invariant Every owned address remains stable for the object lifetime.
    Phase2SealingArena::Phase2SealingArena(
        Phase2EvaluationCapacity fixedCapacity)
        : capacity_(fixedCapacity),
          firingRecords_(fixedCapacity.firingRecords),
          generatedBytes_(fixedCapacity.generatedBytes),
          levelValues_(fixedCapacity.levelValues),
          originDependencies_(fixedCapacity.originDependencies),
          markerKeys_(fixedCapacity.markerKeys),
          markerRemainingArgs_(fixedCapacity.markerRemainingArgs),
          markerArgs_(fixedCapacity.markerArgs),
          firingOrder_(fixedCapacity.firingRecords),
          sealedDependenciesScratch_(fixedCapacity.originDependencies) {
        assert(capacity_.logicalBlocks > 0);
        assert(capacity_.firingRecords > 0);
        assert(capacity_.generatedBytes > 0);
        assert(capacity_.levelValues > 0);
        assert(capacity_.originDependencies > 0);
        assert(capacity_.markerKeys > 0);
        assert(capacity_.markerRemainingArgs > 0);
        assert(capacity_.markerArgs > 0);
        const uint32_t stringScratch = std::max({
            capacity_.markerKeys,
            capacity_.markerRemainingArgs,
            capacity_.markerArgs });
        sealedStringsScratch_.resize(stringScratch);
    }

    /// @brief Download canonical retained GPU records and seal them by block.
    ///
    /// @details
    /// Copies the used device columns into fixed host prefixes, walks only the
    /// doom-compacted canonical permutation, and translates identifiers and byte
    /// slices without reopening any semantic decision. Source implication
    /// dependency zero is decoded through the projected rule interner; every
    /// other dependency and every validity is decoded through the projected
    /// NameMap, exactly matching the processor producer's sealed payload.
    ///
    /// @param evaluation Device evaluator containing the completed sweep.
    /// @param projection Host twin of the projection uploaded for that sweep.
    /// @param used Exact materialization counts returned by the evaluator.
    /// @param retainedFiringCount Exact count returned by doom selection.
    /// @param outputs One bound, filling `SealedPageSet` per logical block.
    /// @param outputCount Number of output sets.
    /// @return Number of appended `FiringRecord`s.
    /// @invariant Retained indices are canonical and grouped by logical block.
    uint32_t Phase2SealingArena::downloadAndSeal(
        const CudaPhase2EvaluationBuffer& evaluation,
        const Phase2ProjectionArena& projection,
        const Phase2FiringExpressionResult& used,
        uint32_t retainedFiringCount,
        SealedPageSet* const* outputs,
        uint32_t outputCount) {
        assert(outputCount == projection.logicalBlocks.size());
        assert(outputCount > 0);
        assert(outputCount <= capacity_.logicalBlocks);
        assert(outputs != nullptr);
        for (uint32_t block = 0; block < outputCount; ++block) {
            assert(outputs[block] != nullptr);
            assert(outputs[block]->filling());
        }
        assert(used.firingRecordCount <= capacity_.firingRecords);
        assert(used.generatedByteCount <= capacity_.generatedBytes);
        assert(used.levelValueCount <= capacity_.levelValues);
        assert(used.originDependencyCount <= capacity_.originDependencies);
        assert(used.markerKeyCount <= capacity_.markerKeys);
        assert(used.markerRemainingArgCount <= capacity_.markerRemainingArgs);
        assert(used.markerArgCount <= capacity_.markerArgs);
        assert(retainedFiringCount <= used.firingRecordCount);

        const uint32_t firingRecordCount = evaluation.downloadFiringRecords(
            firingRecords_.data(), capacity_.firingRecords);
        const uint32_t generatedByteCount = evaluation.downloadGeneratedBytes(
            generatedBytes_.data(), capacity_.generatedBytes);
        const uint32_t levelValueCount = evaluation.downloadLevelValues(
            levelValues_.data(), capacity_.levelValues);
        const uint32_t originDependencyCount =
            evaluation.downloadOriginDependencies(
                originDependencies_.data(), capacity_.originDependencies);
        const uint32_t markerKeyCount = evaluation.downloadMarkerKeys(
            markerKeys_.data(), capacity_.markerKeys);
        const uint32_t markerRemainingArgCount =
            evaluation.downloadMarkerRemainingArgs(
                markerRemainingArgs_.data(), capacity_.markerRemainingArgs);
        const uint32_t markerArgCount = evaluation.downloadMarkerArgs(
            markerArgs_.data(), capacity_.markerArgs);
        const uint32_t firingOrderCount = evaluation.downloadFiringOrder(
            firingOrder_.data(), capacity_.firingRecords);
        assert(firingRecordCount == used.firingRecordCount);
        assert(generatedByteCount == used.generatedByteCount);
        assert(levelValueCount == used.levelValueCount);
        assert(originDependencyCount == used.originDependencyCount);
        assert(markerKeyCount == used.markerKeyCount);
        assert(markerRemainingArgCount == used.markerRemainingArgCount);
        assert(markerArgCount == used.markerArgCount);
        assert(firingOrderCount == retainedFiringCount);

        const auto checkedSlice = [&](DeviceEvaluationByteSlice slice)
            -> StrSpan {
            assert(slice.offset <= used.generatedByteCount);
            assert(slice.length <= used.generatedByteCount - slice.offset);
            assert(slice.length <= static_cast<uint32_t>(
                std::numeric_limits<int32_t>::max()));
            return StrSpan(
                generatedBytes_.data() + slice.offset,
                static_cast<int32_t>(slice.length));
        };
        const auto projectedName = [&](uint32_t blockIndex, NameId id)
            -> StrSpan {
            assert(blockIndex < projection.logicalBlocks.size());
            const DeviceLogicalBlockProjection& block =
                projection.logicalBlocks[blockIndex];
            assert(id > 0);
            assert(static_cast<uint32_t>(id) < block.nameRecordCount);
            const DeviceNameRecord& name = projection.nameRecords[
                block.nameRecordOffset + static_cast<uint32_t>(id)];
            assert(name.byteOffset >= block.nameByteOffset);
            assert(name.byteOffset <= block.nameByteOffset + block.nameByteCount);
            assert(name.byteLength <= block.nameByteOffset + block.nameByteCount
                - name.byteOffset);
            assert(name.byteLength <= static_cast<uint32_t>(
                std::numeric_limits<int32_t>::max()));
            return StrSpan(
                projection.nameBytes.data() + name.byteOffset,
                static_cast<int32_t>(name.byteLength));
        };
        const auto projectedRule = [&](uint32_t blockIndex, int32_t id)
            -> StrSpan {
            assert(blockIndex < projection.logicalBlocks.size());
            const DeviceLogicalBlockProjection& block =
                projection.logicalBlocks[blockIndex];
            assert(id > 0);
            assert(static_cast<uint32_t>(id) < block.ruleStringRecordCount);
            const DeviceRuleStringRecord& rule = projection.ruleStringRecords[
                block.ruleStringRecordOffset + static_cast<uint32_t>(id)];
            assert(rule.byteOffset >= block.ruleStringByteOffset);
            assert(rule.byteOffset
                <= block.ruleStringByteOffset + block.ruleStringByteCount);
            assert(rule.byteLength
                <= block.ruleStringByteOffset + block.ruleStringByteCount
                    - rule.byteOffset);
            assert(rule.byteLength <= static_cast<uint32_t>(
                std::numeric_limits<int32_t>::max()));
            return StrSpan(
                projection.ruleStringBytes.data() + rule.byteOffset,
                static_cast<int32_t>(rule.byteLength));
        };
        const auto seal = [](SealedPageSet& pages, StrSpan bytes) {
            return SealedString::copyFrom(pages, bytes.ptr, bytes.len);
        };

        uint32_t previousBlock = 0;
        bool havePreviousBlock = false;
        for (uint32_t orderIndex = 0;
             orderIndex < retainedFiringCount; ++orderIndex) {
            const uint32_t recordIndex = firingOrder_[orderIndex];
            assert(recordIndex < used.firingRecordCount);
            const DevicePhase2FiringRecord& source =
                firingRecords_[recordIndex];
            assert(source.logicalBlockIndex < outputCount);
            if (havePreviousBlock)
                assert(previousBlock <= source.logicalBlockIndex);
            previousBlock = source.logicalBlockIndex;
            havePreviousBlock = true;
            SealedPageSet& pages = *outputs[source.logicalBlockIndex];

            const bool marker =
                (source.flags & deviceFiringMarker) != 0;
            const bool demand =
                (source.flags & deviceFiringOrdis2Demand) != 0;
            assert(!(marker && demand));
            FiringRecord record;
            record.isMarker = marker;
            record.isOrdis2Demand = demand;
            record.rplExpr2 = seal(pages, checkedSlice(source.expression));
            record.validityName = seal(
                pages, projectedName(
                    source.logicalBlockIndex, source.validityId));

            if (demand) {
                assert(source.levelsOffset <= used.levelValueCount);
                assert(source.levelsCount
                    <= used.levelValueCount - source.levelsOffset);
                record.levels = SealedSpan<int>::copyFrom(
                    pages, levelValues_.data() + source.levelsOffset,
                    static_cast<int32_t>(source.levelsCount));
                assert(source.demandSourceImplId > 0);
                record.demandSourceImplId = source.demandSourceImplId;
            }
            else if (!marker) {
                assert(source.levelsOffset <= used.levelValueCount);
                assert(source.levelsCount
                    <= used.levelValueCount - source.levelsOffset);
                record.levels = SealedSpan<int>::copyFrom(
                    pages, levelValues_.data() + source.levelsOffset,
                    static_cast<int32_t>(source.levelsCount));
                record.originTag = seal(pages, StrSpan("implication", 11));
                assert(source.originDependencyOffset
                    <= used.originDependencyCount);
                assert(source.originDependencyCount
                    <= used.originDependencyCount
                        - source.originDependencyOffset);
                assert(source.originDependencyCount > 0);
                assert(source.originDependencyCount
                    <= sealedDependenciesScratch_.size());
                for (uint32_t dependency = 0;
                     dependency < source.originDependencyCount; ++dependency) {
                    const DeviceEvaluationDependency& input =
                        originDependencies_[
                            source.originDependencyOffset + dependency];
                    const StrSpan original = dependency == 0
                        ? projectedRule(source.logicalBlockIndex,
                              static_cast<int32_t>(input.originalId))
                        : projectedName(source.logicalBlockIndex,
                              input.originalId);
                    sealedDependenciesScratch_[dependency] =
                        SealedExpressionWithValidity{
                            seal(pages, original),
                            seal(pages, projectedName(
                                source.logicalBlockIndex,
                                input.validityId)) };
                }
                record.originDeps = SealedSpan<
                    SealedExpressionWithValidity>::copyFrom(
                        pages, sealedDependenciesScratch_.data(),
                        static_cast<int32_t>(
                            source.originDependencyCount));
                record.doNotDisintegrate = (source.flags
                    & deviceFiringDoNotDisintegrate) != 0;
                record.allowOrDisintegration = (source.flags
                    & deviceFiringAllowOrDisintegration) != 0;
                record.allGood =
                    (source.flags & deviceFiringAllGood) != 0;
                record.alreadyKnown =
                    (source.flags & deviceFiringAlreadyKnown) != 0;
                record.iteration = source.iteration;
            }
            else {
                assert(source.markerArgOffset <= used.markerArgCount);
                assert(source.markerArgCount
                    <= used.markerArgCount - source.markerArgOffset);
                assert(source.markerArgCount <= sealedStringsScratch_.size());
                for (uint32_t argument = 0;
                     argument < source.markerArgCount; ++argument) {
                    sealedStringsScratch_[argument] = seal(
                        pages, checkedSlice(markerArgs_[
                            source.markerArgOffset + argument]));
                }
                record.markerArgsSorted = SealedSpan<SealedString>::copyFrom(
                    pages, sealedStringsScratch_.data(),
                    static_cast<int32_t>(source.markerArgCount));

                assert(source.markerKeyOffset <= used.markerKeyCount);
                assert(source.markerKeyCount
                    <= used.markerKeyCount - source.markerKeyOffset);
                assert(source.markerKeyCount <= sealedStringsScratch_.size());
                for (uint32_t key = 0; key < source.markerKeyCount; ++key) {
                    sealedStringsScratch_[key] = seal(
                        pages, checkedSlice(markerKeys_[
                            source.markerKeyOffset + key]));
                }
                record.admv.key = SealedSpan<SealedString>::copyFrom(
                    pages, sealedStringsScratch_.data(),
                    static_cast<int32_t>(source.markerKeyCount));

                assert(source.markerRemainingArgOffset
                    <= used.markerRemainingArgCount);
                assert(source.markerRemainingArgCount
                    <= used.markerRemainingArgCount
                        - source.markerRemainingArgOffset);
                assert(source.markerRemainingArgCount
                    <= sealedStringsScratch_.size());
                for (uint32_t argument = 0;
                     argument < source.markerRemainingArgCount; ++argument) {
                    sealedStringsScratch_[argument] = seal(
                        pages, projectedRule(
                            source.logicalBlockIndex,
                            markerRemainingArgs_[
                                source.markerRemainingArgOffset + argument]));
                }
                record.admv.remainingArgsSorted =
                    SealedSpan<SealedString>::copyFrom(
                        pages, sealedStringsScratch_.data(),
                        static_cast<int32_t>(
                            source.markerRemainingArgCount));
                record.admv.standardMaxAdmissionDepth =
                    source.standardMaxAdmissionDepth;
                record.admv.standardMaxSecondaryNumber =
                    source.standardMaxSecondaryNumber;
                record.admv.flag = false;
                record.admv.ordisOnly =
                    (source.flags & deviceFiringOrdisOnly) != 0;
                record.markerNotAtomic =
                    (source.flags & deviceFiringMarkerNotAtomic) != 0;
            }
            pages.appendRecord(record);
        }
        for (uint32_t block = 0; block < outputCount; ++block)
            outputs[block]->seal();
        return retainedFiringCount;
    }

} // namespace gl::gpu

#endif  // GL_CUDA
