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
#include "../infra/phase2_trace.hpp"
#include <optional>

#include <algorithm>
#include <cassert>
#include <chrono>
#include <limits>

namespace gl::gpu {

    static_assert(deviceIntegrationJustification
        == static_cast<int32_t>(RuleJustification::integration));
    static_assert(deviceGoalDemandJustification
        == static_cast<int32_t>(RuleJustification::goalDemand));

    /// @brief Measure all fixed download and ownership-conversion scratch.
    /// @details Columns remain live together until this chunk finishes sealing.
    /// @param capacity Positive evaluator capacity.
    /// @return Exact aligned payload layout before block rounding.
    GpuHostLayout Phase2SealingArena::hostLayout(Phase2EvaluationCapacity capacity) {
        GpuHostLayout layout;
        layout.add<DevicePhase2FiringRecord>(capacity.firingRecords);
        layout.add<char>(capacity.generatedBytes);
        layout.add<int32_t>(capacity.levelValues);
        layout.add<DeviceEvaluationDependency>(capacity.originDependencies);
        layout.add<DeviceEvaluationByteSlice>(capacity.markerKeys);
        layout.add<int32_t>(capacity.markerRemainingArgs);
        layout.add<DeviceEvaluationByteSlice>(capacity.markerArgs);
        layout.add<uint32_t>(capacity.firingRecords);
        layout.add<SealedString>(std::max({capacity.markerKeys, capacity.markerRemainingArgs, capacity.markerArgs}));
        layout.add<SealedExpressionWithValidity>(capacity.originDependencies);
        return layout;
    }

    /// @brief Record evaluator capacity for a measured first-download lease.
    ///
    /// @details
    /// Columns remain unallocated through construction and upload. The first
    /// download reserves the measured lease; later windows reuse its columns.
    ///
    /// @param fixedCapacity Immutable evaluator element and byte ceilings.
    /// @return An empty reusable sealing arena.
    /// @invariant No column grows beyond the evaluator capacity.
    Phase2SealingArena::Phase2SealingArena(
        Phase2EvaluationCapacity fixedCapacity)
        : capacity_(fixedCapacity) {
        assert(capacity_.logicalBlocks > 0);
        assert(capacity_.firingRecords > 0);
        assert(capacity_.generatedBytes > 0);
        assert(capacity_.levelValues > 0);
        assert(capacity_.originDependencies > 0);
        assert(capacity_.markerKeys > 0);
        assert(capacity_.markerRemainingArgs > 0);
        assert(capacity_.markerArgs > 0);

    }

    /// @brief Reserve measured output columns at the first download.
    /// @details Completed construction windows have returned their leases;
    /// subsequent downloads reuse these fixed columns without allocator growth.
    /// @return Nothing.
    void Phase2SealingArena::reserveStorage() {
        assert(hostStorage_.reservedBytes() == 0);
        hostStorage_.reserve(hostLayout(capacity_).bytes);
        firingRecords_.bindStorage(hostStorage_);
        firingRecords_.reserve(capacity_.firingRecords);
        generatedBytes_.bindStorage(hostStorage_);
        generatedBytes_.reserve(capacity_.generatedBytes);
        levelValues_.bindStorage(hostStorage_);
        levelValues_.reserve(capacity_.levelValues);
        originDependencies_.bindStorage(hostStorage_);
        originDependencies_.reserve(capacity_.originDependencies);
        markerKeys_.bindStorage(hostStorage_);
        markerKeys_.reserve(capacity_.markerKeys);
        markerRemainingArgs_.bindStorage(hostStorage_);
        markerRemainingArgs_.reserve(capacity_.markerRemainingArgs);
        markerArgs_.bindStorage(hostStorage_);
        markerArgs_.reserve(capacity_.markerArgs);
        firingOrder_.bindStorage(hostStorage_);
        firingOrder_.reserve(capacity_.firingRecords);
        sealedStringsScratch_.bindStorage(hostStorage_);
        sealedStringsScratch_.reserve(std::max({capacity_.markerKeys, capacity_.markerRemainingArgs, capacity_.markerArgs}));
        sealedDependenciesScratch_.bindStorage(hostStorage_);
        sealedDependenciesScratch_.reserve(capacity_.originDependencies);
    }

    /// @brief Download retained GPU records into their owning block pages.
    ///
    /// @details
    /// Copies the used device columns into fixed host prefixes, walks only the
    /// selected raw or canonical permutation, and translates identifiers and byte
    /// slices without reopening any semantic decision. Source implication
    /// dependency zero is decoded through the projected rule interner; every
    /// other dependency and every validity is decoded through the projected
    /// NameMap, exactly matching the processor producer's sealed payload.
    ///
    /// @param evaluation Device evaluator containing the completed sweep.
    /// @param projection Host twin of the projection uploaded for that sweep.
    /// @param used Exact materialization counts returned by the evaluator.
    /// @param retainedFiringCount Number of records retained in this window.
    /// @param outputs One bound, filling `SealedPageSet` per logical block.
    /// @param outputCount Number of output sets.
    /// @param options Ordering, page-set completion and fragment ownership.
    /// @return Number of appended `FiringRecord`s.
    /// @invariant Canonical indices, when selected, are grouped by logical block.
    uint32_t Phase2SealingArena::downloadAndSeal(
        const CudaPhase2EvaluationBuffer& evaluation,
        const Phase2ProjectionArena& projection,
        const Phase2FiringExpressionResult& used,
        uint32_t retainedFiringCount,
        SealedPageSet* const* outputs,
        uint32_t outputCount,
        Phase2SealingOptions options) {
        gl::Phase2TraceScope phase2OperationTrace("Phase2SealingArena::downloadAndSeal");
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
        assert(options.receive == nullptr || !options.finishOutputs);
        assert(options.canonicalOrder || retainedFiringCount == used.firingRecordCount);
        assert(options.repeatedMarkerHeader == nullptr
            || (retainedFiringCount == 1 && options.receive != nullptr
                && options.repeatedMarkerHeader->isMarker));

        if (hostStorage_.reservedBytes() == 0) reserveStorage();
        // Used prefixes change per download; all backing was budgeted once.
        const auto growTo = [](auto& column, uint32_t count) {
            if (column.size() < count) column.resize(count);
        };
        growTo(firingRecords_, used.firingRecordCount);
        growTo(generatedBytes_, used.generatedByteCount);
        growTo(levelValues_, used.levelValueCount);
        growTo(originDependencies_, used.originDependencyCount);
        growTo(markerKeys_, used.markerKeyCount);
        growTo(markerRemainingArgs_, used.markerRemainingArgCount);
        growTo(markerArgs_, used.markerArgCount);
        if (options.canonicalOrder) growTo(firingOrder_, retainedFiringCount);
        // One record's dependencies and strings are a slice of the window's
        // columns, so the window's counts bound the per-record scratch.
        growTo(sealedDependenciesScratch_, used.originDependencyCount);
        growTo(sealedStringsScratch_, std::max({ used.markerKeyCount,
            used.markerRemainingArgCount, used.markerArgCount }));
        const auto columnCapacity = [](const auto& column) {
            return static_cast<uint32_t>(column.size());
        };

        const uint32_t firingRecordCount = evaluation.downloadFiringRecords(
            firingRecords_.data(), columnCapacity(firingRecords_));
        const uint32_t generatedByteCount = evaluation.downloadGeneratedBytes(
            generatedBytes_.data(), columnCapacity(generatedBytes_));
        const uint32_t levelValueCount = evaluation.downloadLevelValues(
            levelValues_.data(), columnCapacity(levelValues_));
        const uint32_t originDependencyCount =
            evaluation.downloadOriginDependencies(
                originDependencies_.data(), columnCapacity(originDependencies_));
        const uint32_t markerKeyCount = evaluation.downloadMarkerKeys(
            markerKeys_.data(), columnCapacity(markerKeys_));
        const uint32_t markerRemainingArgCount =
            evaluation.downloadMarkerRemainingArgs(
                markerRemainingArgs_.data(), columnCapacity(markerRemainingArgs_));
        const uint32_t markerArgCount = evaluation.downloadMarkerArgs(
            markerArgs_.data(), columnCapacity(markerArgs_));
        const uint32_t firingOrderCount = options.canonicalOrder
            ? evaluation.downloadFiringOrder(firingOrder_.data(), columnCapacity(firingOrder_))
            : retainedFiringCount;
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
        uint32_t sealedRecords = 0;
        for (uint32_t orderIndex = 0;
             orderIndex < retainedFiringCount; ++orderIndex) {
            const uint32_t recordIndex = options.canonicalOrder ? firingOrder_[orderIndex] : orderIndex;
            assert(recordIndex < used.firingRecordCount);
            const DevicePhase2FiringRecord& source =
                firingRecords_[recordIndex];
            assert(source.logicalBlockIndex < outputCount);
            if (options.blockStops != nullptr) {
                auto& stop = options.blockStops[source.logicalBlockIndex];
                assert(stop == INT64_MAX || stop == phase2PendingBlockStop || stop == 0);
                if (stop == 0) continue;
                if ((source.flags & deviceFiringDoomTrigger) != 0)
                    stop = phase2PendingBlockStop;
            }
            ++sealedRecords;
            if (havePreviousBlock && options.canonicalOrder)
                assert(previousBlock <= source.logicalBlockIndex);
            previousBlock = source.logicalBlockIndex;
            havePreviousBlock = true;
            SealedPageSet& pages = *outputs[source.logicalBlockIndex];

            const bool marker =
                (source.flags & deviceFiringMarker) != 0;
            const bool demand =
                (source.flags & deviceFiringOrdis2Demand) != 0;
            assert(!(marker && demand));
            assert(options.repeatedMarkerHeader == nullptr || marker);
            const bool goalNearMiss =
                (source.flags & deviceFiringGoalNearMiss) != 0;
            assert(!goalNearMiss || demand);
            FiringRecord record;
            record.isMarker = marker;
            record.isOrdis2Demand = demand;
            record.isGoalNearMiss = goalNearMiss;
            if (options.repeatedMarkerHeader) {
                record = *options.repeatedMarkerHeader;
                const StrSpan expression = checkedSlice(source.expression);
                const StrSpan validity = projectedName(source.logicalBlockIndex, source.validityId);
                assert(record.rplExpr2.size() == expression.len);
                assert(std::memcmp(record.rplExpr2.data(), expression.ptr, expression.len) == 0);
                assert(record.validityName.size() == validity.len);
                assert(std::memcmp(record.validityName.data(), validity.ptr, validity.len) == 0);
                assert(record.admv.standardMaxAdmissionDepth == source.standardMaxAdmissionDepth);
                assert(record.admv.standardMaxSecondaryNumber == source.standardMaxSecondaryNumber);
            }
            else {
                record.rplExpr2 = seal(pages, checkedSlice(source.expression));
                record.validityName = seal(pages,
                    projectedName(source.logicalBlockIndex, source.validityId));
            }

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
                record.recursionHead = (source.flags
                    & deviceFiringRecursionHead) != 0;
                record.alreadyKnown =
                    (source.flags & deviceFiringAlreadyKnown) != 0;
                record.iteration = source.iteration;
            }
            else {
                if (options.repeatedMarkerHeader == nullptr) {
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
                }

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

                if (options.repeatedMarkerHeader == nullptr) {
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
                }
            }
            if (options.receive) options.receive(options.context, source.logicalBlockIndex, record, source);
            else pages.appendRecord(record);
        }
        if (options.finishOutputs) {
            for (uint32_t block = 0; block < outputCount; ++block)
                outputs[block]->seal();
        }
        return sealedRecords;
    }

    /// @brief Execute a complete GPU request stream through bounded downloads.
    ///
    /// @details
    /// Freezes generation at each request drain, evaluates every request
    /// through bounded owner/value windows, and copies
    /// complete results to host-owned pages. Marker keys assemble before a
    /// record is published. No inference or substitution runs on the host.
    /// A CUDA discharge retains the complete current growth level and stops
    /// subsequent levels for that block. Capacity drains never commit a stop.
    /// Output pages seal only after generation and evaluation are exhausted.
    ///
    /// @param buffers Device owners holding the immutable input and scratch.
    /// @param projection Host twin of the uploaded projection.
    /// @param tasks Host twin of uploaded original task identities.
    /// @param schedule Original call schedule used by generation.
    /// @param parameters Existing GPU growth policy.
    /// @param plan Allocated processing capacities for occupancy measurement.
    /// @param outputs Bound filling pages for each projected logical block.
    /// @param outputCount Number of projected blocks and output page sets.
    /// @param taskSubkeyCounts Complete cumulative subkey counts per task.
    /// @param taskCapacity Number of supplied task counter slots.
    /// @param doomLines Optional per-block stop states; returns zero or INT64_MAX.
    /// @param frontierStorage Prepared request-frontier files and resident budget.
    /// @return Logical totals, transfer bytes and completed storage pressure.
    /// @invariant Scratch fits one CPU-supported output fragment; snapshots
    ///            stay immutable and no statement is deposited during drains.
    Phase2StreamExecutionResult Phase2SealingArena::runWindows(
        Phase2StreamBuffers& buffers, const Phase2ProjectionArena& projection,
        const Phase2TaskProjectionArena& tasks, const Phase2GrowthScheduleArena& schedule,
        const DevicePhase2GrowthParameters& parameters, const Phase2BufferPlan& plan,
        SealedPageSet* const* outputs, uint32_t outputCount,
        uint64_t* taskSubkeyCounts, uint32_t taskCapacity, int64_t* doomLines, RequestFrontierStorageOptions frontierStorage) {
        gl::Phase2TraceScope phase2OperationTrace("Phase2SealingArena::runWindows");
        assert(outputCount == projection.logicalBlocks.size());
        for (uint32_t block = 0; block < outputCount; ++block)
            assert(outputs[block] != nullptr && outputs[block]->filling());
        Phase2StreamExecutionResult result{};
        if (doomLines != nullptr)
            std::fill(doomLines, doomLines + outputCount, INT64_MAX);
        struct Context {
            Phase2SealingArena& sealing;
            Phase2StreamBuffers& buffers;
            const Phase2ProjectionArena& projection;
            SealedPageSet* const* outputs;
            uint32_t outputCount;
            const Phase2BufferPlan& plan;
            Phase2StreamExecutionResult& result;
            const Phase2TaskProjectionArena& tasks;
            const Phase2GrowthScheduleArena& schedule;
            int64_t* blockStops;
        } context{*this, buffers, projection, outputs, outputCount, plan, result,
            tasks, schedule, doomLines};
        const auto drain = [](void* opaque, const CudaPhase2GrowthBuffer&, bool levelComplete) {
            gl::Phase2TraceScope phase2DrainTrace("GL.Phase2.RequestDrain", levelComplete);
            auto& context = *static_cast<Context*>(opaque);
            auto& buffers = context.buffers;
            auto& result = context.result;
            const auto ordered = buffers.ordering.orderRequests(
                buffers.projection, buffers.tasks, buffers.growth);
            result.feedback.occupied[0] = std::max(result.feedback.occupied[0],
                static_cast<uint64_t>(ordered.orderedEventCount));
            result.feedback.occupied[1] = std::max(result.feedback.occupied[1],
                static_cast<uint64_t>(ordered.requestCount));
            const uint32_t requestCount = ordered.requestCount;
            result.evaluatedRequests += requestCount;
            Phase2EvaluationStreamMetrics metrics{};
            buffers.evaluation.beginEvaluationWindows(buffers.projection, buffers.tasks,
                buffers.growth, buffers.ordering, &metrics);
            while (buffers.evaluation.nextEvaluationValues(buffers.projection,
                buffers.tasks, buffers.growth, buffers.ordering)) {
                Phase2FiringCursor cursor{};
                struct Marker {
                    SealedPageSet* const* outputs;
                    uint32_t totalKeys{0};
                    uint32_t block{UINT32_MAX};
                    FiringRecord record{};
                    std::optional<SealedSpanBuilder<SealedString>> keys;
                } marker{context.outputs};
                for (;;) {
                    const auto window = buffers.evaluation.nextFiringWindow(
                        buffers.projection, buffers.tasks, buffers.growth,
                        buffers.ordering, cursor);
                    if (window.state == Phase2FiringWindowState::complete) break;
                    assert(window.state == Phase2FiringWindowState::ready
                        && "allocated output minimum must cover one CPU-supported fragment");
                    result.feedback.occupied[7] = std::max(result.feedback.occupied[7],
                        static_cast<uint64_t>(window.valueBegin) + window.valueCount);
                    const auto& used = window.used;
                    const uint64_t columns[7] = {used.firingRecordCount,
                        used.generatedByteCount, used.levelValueCount,
                        used.originDependencyCount, used.markerKeyCount,
                        used.markerRemainingArgCount, used.markerArgCount};
                    const uint64_t bytesPerRow[7] = {sizeof(DevicePhase2FiringRecord), 1,
                        sizeof(int32_t), sizeof(DeviceEvaluationDependency),
                        sizeof(DeviceEvaluationByteSlice), sizeof(int32_t),
                        sizeof(DeviceEvaluationByteSlice)};
                    for (uint32_t column = 0; column < 7; ++column) {
                        result.outputColumns[column] += columns[column];
                        result.outputTransferBytes += columns[column] * bytesPerRow[column];
                        result.feedback.occupied[8 + column] = std::max(
                            result.feedback.occupied[8 + column], columns[column]);
                    }
                    Phase2SealingOptions options{false, false};
                    if (context.blockStops != nullptr) {
                        buffers.evaluation.markDoomTriggers(buffers.projection);
                        options.blockStops = context.blockStops;
                    }
                    if (window.markerFragment) {
                        options.blockStops = nullptr;
                        marker.totalKeys = window.totalMarkerKeys;
                        options.context = &marker;
                        options.repeatedMarkerHeader = marker.keys ? &marker.record : nullptr;
                        options.receive = [](void* opaque, uint32_t block, const FiringRecord& record,
                            const DevicePhase2FiringRecord&) {
                            auto& marker = *static_cast<Marker*>(opaque);
                            assert(record.isMarker);
                            if (!marker.keys) {
                                marker.block = block;
                                marker.record = record;
                                marker.keys.emplace(*marker.outputs[block], marker.totalKeys);
                            }
                            assert(marker.block == block);
                            marker.keys->append(record.admv.key.data(), record.admv.key.size());
                        };
                    } else {
                        assert(!marker.keys);
                    }
                    const auto sealingStarted = std::chrono::steady_clock::now();
                    const uint32_t downloaded = context.sealing.downloadAndSeal(
                        buffers.evaluation, context.projection, used, used.firingRecordCount,
                        context.outputs, context.outputCount, options);
                    result.sealingSeconds += std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - sealingStarted).count();
                    assert(downloaded <= used.firingRecordCount);
                    assert(options.blockStops != nullptr || downloaded == used.firingRecordCount);
                    result.sealedRecords += downloaded;
                    ++result.outputDownloads;
                    if (window.markerFragment) {
                        assert(marker.keys);
                        if (window.markerKeyBegin + window.markerKeyCount == window.totalMarkerKeys) {
                            marker.record.admv.key = marker.keys->finish();
                            if (context.blockStops == nullptr || context.blockStops[marker.block] != 0) {
                                context.outputs[marker.block]->appendRecord(marker.record);
                                ++result.firingRecords;
                            }
                            marker.keys.reset();
                        }
                    } else result.firingRecords += downloaded;
                    const uint64_t capacities[7] = {context.plan.evaluation.firingRecords,
                        context.plan.evaluation.generatedBytes, context.plan.evaluation.levelValues,
                        context.plan.evaluation.originDependencies, context.plan.evaluation.markerKeys,
                        context.plan.evaluation.markerRemainingArgs, context.plan.evaluation.markerArgs};
                    for (uint32_t column = 0; column < 7; ++column)
                        if (columns[column] == capacities[column]) ++result.feedback.drains[8 + column];
                }
                assert(!marker.keys);
            }
            result.feedback.occupied[7] = std::max(result.feedback.occupied[7],
                static_cast<uint64_t>(std::max(metrics.maximumOwnerWindow, metrics.maximumValueWindow)));
            result.feedback.drains[7] += metrics.ownerWindows + metrics.valueWindows;
            if (levelComplete && context.blockStops != nullptr) {
                for (uint32_t block = 0; block < context.outputCount; ++block)
                    if (context.blockStops[block] == phase2PendingBlockStop)
                        context.blockStops[block] = 0;
            }
        };
        const auto retainedCalls = [](void* opaque, uint8_t* keepByCall, uint32_t callCount) {
            auto& context = *static_cast<Context*>(opaque);
            assert(context.blockStops != nullptr);
            assert(callCount == context.schedule.calls.size());
            bool cutoff = false;
            for (uint32_t call = 0; call < callCount; ++call) {
                const auto taskIndex = context.schedule.calls[call].taskIndex;
                assert(taskIndex < context.tasks.tasks.size());
                const auto block = context.tasks.tasks[taskIndex].logicalBlockIndex;
                assert(block < context.outputCount);
                keepByCall[call] = context.blockStops[block] != 0 ? 1 : 0;
                if (keepByCall[call] == 0) cutoff = true;
            }
            return cutoff;
        };
        const auto generationStarted = std::chrono::steady_clock::now();
        result.growth = buffers.growth.runRequestGrowthWindows(buffers.projection,
            buffers.tasks, tasks, buffers.filter, schedule, parameters, drain, &context,
            taskSubkeyCounts, taskCapacity, doomLines != nullptr ? +retainedCalls : nullptr,
            frontierStorage);
        result.generationAndDrainSeconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - generationStarted).count();
        result.feedback.drains[0] = result.growth.drainCount - result.growth.cutoffDrainCount;
        result.feedback.drains[1] = result.growth.drainCount - result.growth.cutoffDrainCount;
        result.feedback.occupied[2] = std::min(result.growth.maximumFrontierCount,
            static_cast<uint64_t>(plan.growth.frontierRecords));
        // The whole measured level sizes the pass-local frontier store.
        result.feedback.occupied[static_cast<uint32_t>(Phase2WorkDimension::frontierStore)] =
            result.growth.maximumFrontierCount;
        result.feedback.drains[static_cast<uint32_t>(Phase2WorkDimension::frontierStore)] =
            result.growth.frontierSpills;
        result.feedback.occupied[3] = result.growth.maximumPrefixPayloadValues;
        result.feedback.occupied[4] = result.growth.maximumPrefixVariableValues;
        result.feedback.occupied[5] = result.growth.maximumPrefixSecondaryValues;
        result.feedback.occupied[6] = result.growth.maximumCandidateWindow;
        for (uint32_t dimension = 3; dimension < 6; ++dimension)
            result.feedback.drains[dimension] = result.growth.prefixWindows;
        result.feedback.drains[6] = result.growth.candidateWindows;
        if (doomLines != nullptr)
            for (uint32_t block = 0; block < outputCount; ++block)
                assert(doomLines[block] == 0 || doomLines[block] == INT64_MAX);
        for (uint32_t block = 0; block < outputCount; ++block) outputs[block]->seal();
        return result;
    }

} // namespace gl::gpu

#endif  // GL_CUDA
