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
/// @brief Direct unit tests for the Windows CUDA build and runtime contract.
///
/// @details
/// The first test validates the exact device properties required by the first
/// Phase 2 port. The second launches native device code and verifies its copied
/// result, proving more than enumeration or successful linking. The third uses
/// the reusable CUDA-event owner around that real device work and requires a
/// positive device interval. Any CUDA failure asserts in production; there is no
/// processor fallback to conceal it.

#include "test_harness.hpp"

#include "../gpu/phase2_cuda.hpp"
#include "../gpu/phase2_allocation.hpp"
#include "../memory_infra/sealed_pages.hpp"

#include <cstdint>
#include <cuda_runtime.h>

CUDA_TEST(phase2_cuda, device_contract_is_supported) {
    const gl::gpu::CudaDeviceContract contract =
        gl::gpu::queryCudaDeviceContract();
    ASSERT_GE(contract.deviceOrdinal, 0);
    ASSERT_TRUE(contract.computeMajor > 8
        || (contract.computeMajor == 8 && contract.computeMinor >= 9));
    ASSERT_TRUE(contract.totalGlobalBytes > 0);
    ASSERT_TRUE(contract.multiprocessorCount > 0);
    ASSERT_TRUE(contract.maximumThreadsPerBlock >= 256);
}

CUDA_TEST(phase2_cuda, native_kernel_launches_and_round_trips) {
    constexpr uint32_t input = 4070u;
    constexpr uint32_t expected = input * 1664525u + 1013904223u;
    ASSERT_EQ(gl::gpu::launchCudaContractProbe(input), expected);
}

// A compute-engine publication lands exactly the requested words in mapped
// host memory, including at an interior destination, once a later event completes.
CUDA_TEST(phase2_cuda, published_device_words_reach_mapped_host_memory) {
    constexpr uint32_t words = 300;
    uint32_t pattern[words];
    for (uint32_t i = 0; i < words; ++i) pattern[i] = (i + 1) * 2654435761u;
    uint32_t* device = nullptr;
    ASSERT_EQ(cudaMalloc(reinterpret_cast<void**>(&device), sizeof(pattern)), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(device, pattern, sizeof(pattern), cudaMemcpyHostToDevice), cudaSuccess);
    uint32_t* host = nullptr;
    ASSERT_EQ(cudaHostAlloc(reinterpret_cast<void**>(&host), (words + 2) * sizeof(uint32_t),
        cudaHostAllocMapped), cudaSuccess);
    for (uint32_t i = 0; i < words + 2; ++i) host[i] = 0xffffffffu;
    cudaEvent_t done{};
    ASSERT_EQ(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), cudaSuccess);
    gl::gpu::publishDeviceWords(device, words, host + 1);
    ASSERT_EQ(cudaEventRecord(done, 0), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(done), cudaSuccess);
    ASSERT_EQ(host[0], 0xffffffffu);
    for (uint32_t i = 0; i < words; ++i) ASSERT_EQ(host[i + 1], pattern[i]);
    ASSERT_EQ(host[words + 1], 0xffffffffu);
    gl::gpu::publishDeviceWords(device + 17, 1, host);
    ASSERT_EQ(cudaEventRecord(done, 0), cudaSuccess);
    ASSERT_EQ(cudaEventSynchronize(done), cudaSuccess);
    ASSERT_EQ(host[0], pattern[17]);
    ASSERT_EQ(host[1], pattern[0]);
    ASSERT_EQ(cudaEventDestroy(done), cudaSuccess);
    ASSERT_EQ(cudaFreeHost(host), cudaSuccess);
    ASSERT_EQ(cudaFree(device), cudaSuccess);
}

CUDA_TEST(phase2_cuda, reusable_device_timer_measures_native_work) {
    gl::gpu::CudaPhase2DeviceTimer timer;
    timer.start();
    constexpr uint32_t input = 89u;
    constexpr uint32_t expected = input * 1664525u + 1013904223u;
    ASSERT_EQ(gl::gpu::launchCudaContractProbe(input), expected);
    ASSERT_TRUE(timer.stopSeconds() > 0.0);
}

CUDA_TEST(phase2_cuda, automatic_block_pool_reuses_storage_and_coalesces_free_runs) {
    gl::gpu::CudaPhase2BlockPool pool(8ull * 1024 * 1024);
    ASSERT_EQ(pool.budget().blocks, 4u);
    ASSERT_EQ(pool.availableBlocks(), 4u);
    ASSERT_EQ(pool.largestFreeRun(), 4u);
    const auto first = pool.reserveBlocks(1);
    const auto middle = pool.reserveBlocks(2);
    const auto last = pool.reserveBlocks(1);
    ASSERT_EQ(pool.availableBlocks(), 0u);
    ASSERT_EQ(pool.largestFreeRun(), 0u);
    const uint64_t value = 0x123456789abcdef0ull;
    uint64_t output = 0;
    ASSERT_EQ(cudaMemcpy(pool.address(middle), &value, sizeof(value), cudaMemcpyHostToDevice), cudaSuccess);
    ASSERT_EQ(cudaMemcpy(&output, pool.address(middle), sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
    ASSERT_EQ(output, value);
    pool.release(first);
    pool.release(last);
    ASSERT_EQ(pool.availableBlocks(), 2u);
    ASSERT_EQ(pool.largestFreeRun(), 1u);
    const auto replacement = pool.reserveBlocks(1);
    ASSERT_EQ(replacement.first, first.first);
    ASSERT_NE(replacement.generation, first.generation);
    pool.release(replacement);
    pool.release(middle);
    ASSERT_EQ(pool.availableBlocks(), 4u);
    ASSERT_EQ(pool.largestFreeRun(), 4u);
}

CUDA_TEST(phase2_cuda, leased_arena_aligns_arrays_and_returns_its_complete_run) {
    gl::gpu::CudaPhase2BlockPool pool(8ull * 1024 * 1024);
    {
        gl::gpu::CudaPhase2Arena arena(pool, 2);
        void* first = nullptr;
        void* second = nullptr;
        arena.allocate(&first, 10);
        arena.allocate(&second, 16);
        ASSERT_EQ(reinterpret_cast<uintptr_t>(second) - reinterpret_cast<uintptr_t>(first), 256ull);
        ASSERT_EQ(arena.usedBytes(), 272ull);
        ASSERT_EQ(arena.capacityBytes(), 4ull * 1024 * 1024);
        ASSERT_EQ(pool.availableBlocks(), 2u);
        const uint64_t input = 0xabcdef0123456789ull;
        uint64_t output = 0;
        ASSERT_EQ(cudaMemcpy(second, &input, sizeof(input), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&output, second, sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(output, input);
    }
    ASSERT_EQ(pool.availableBlocks(), 4u);
    ASSERT_EQ(pool.largestFreeRun(), 4u);
}

CUDA_TEST(phase2_cuda, all_buffer_owners_borrow_one_lease_without_freeing_its_arrays) {
    gl::gpu::CudaPhase2BlockPool pool(16ull * 1024 * 1024);
    {
        gl::gpu::CudaPhase2Arena arena(pool, 4);
        {
            uint64_t measuredCursor = 0;
            const auto verifySizing = [&](const auto& buffer, gl::gpu::Phase2DeviceAllocationSize size) {
                ASSERT_EQ(size.deviceBytes, buffer.fixedAllocationBytes());
                measuredCursor = ((measuredCursor + 255) & ~uint64_t{255}) + size.arenaBytes;
                ASSERT_EQ(measuredCursor, arena.usedBytes());
            };
            const gl::gpu::Phase2ProjectionCapacity projectionCapacity{
                1, 8, 256, 16384, 1024, 256, 16384, 10, 1024, 4096,
                65536, 1024, 65536, 1, 1024, 4096, 65536, 1024,
                11, 1024, 4096, 1024, 32, 4096};
            gl::gpu::CudaPhase2ProjectionBuffer projection(projectionCapacity, &arena);
            verifySizing(projection, gl::gpu::CudaPhase2ProjectionBuffer::measureAllocation(projectionCapacity));
            const gl::gpu::Phase2TaskProjectionCapacity taskCapacity{2, 8, 8, 8};
            gl::gpu::CudaPhase2TaskBuffer tasks(taskCapacity, &arena);
            verifySizing(tasks, gl::gpu::CudaPhase2TaskBuffer::measureAllocation(taskCapacity));
            const gl::gpu::Phase2FilterScheduleCapacity filterCapacity{8, 128, 64, 32};
            gl::gpu::CudaPhase2FilterSortBuffer filter(filterCapacity, &arena);
            verifySizing(filter, gl::gpu::CudaPhase2FilterSortBuffer::measureAllocation(filterCapacity));
            const gl::gpu::Phase2GrowthCapacity growthCapacity{8, 64, 16, 32, 16, 512, 256, 64, 32, 16};
            gl::gpu::CudaPhase2GrowthBuffer growth(growthCapacity, &arena);
            verifySizing(growth, gl::gpu::CudaPhase2GrowthBuffer::measureAllocation(growthCapacity));
            const gl::gpu::Phase2OrderingCapacity orderingCapacity{32, 16};
            gl::gpu::CudaPhase2OrderingBuffer ordering(orderingCapacity, &arena);
            verifySizing(ordering, gl::gpu::CudaPhase2OrderingBuffer::measureAllocation(orderingCapacity));
            const gl::gpu::Phase2EvaluationCapacity evaluationCapacity{
                1, 16, 16, 16, 16, 16, 16, 4096, 128, 128, 16, 32, 32};
            gl::gpu::CudaPhase2EvaluationBuffer evaluation(evaluationCapacity, &arena);
            verifySizing(evaluation, gl::gpu::CudaPhase2EvaluationBuffer::measureAllocation(evaluationCapacity));
            ASSERT_TRUE(arena.usedBytes() > 0);
            ASSERT_TRUE(arena.usedBytes() < arena.capacityBytes());
            ASSERT_EQ(pool.availableBlocks(), 4u);
        }
        // Buffer destruction leaves the arena lease alive for its owner.
        ASSERT_EQ(pool.availableBlocks(), 4u);
        void* word = nullptr;
        arena.allocate(&word, sizeof(uint32_t));
        const uint32_t input = 123456789u;
        uint32_t output = 0;
        ASSERT_EQ(cudaMemcpy(word, &input, sizeof(input), cudaMemcpyHostToDevice), cudaSuccess);
        ASSERT_EQ(cudaMemcpy(&output, word, sizeof(output), cudaMemcpyDeviceToHost), cudaSuccess);
        ASSERT_EQ(input, output);
    }
    ASSERT_EQ(pool.availableBlocks(), 8u);
}

CUDA_TEST(phase2_cuda, block_pool_reports_device_totals_for_the_memory_census) {
    gl::gpu::CudaPhase2BlockPool pool(8ull * 1024 * 1024);
    ASSERT_TRUE(pool.deviceTotalBytes() > 0);
    ASSERT_TRUE(pool.deviceFreeBeforePoolBytes() > 0);
    ASSERT_TRUE(pool.deviceFreeBeforePoolBytes() <= pool.deviceTotalBytes());
    ASSERT_TRUE(pool.budget().bytes <= pool.deviceFreeBeforePoolBytes());
}

CUDA_TEST(phase2_cuda, projection_buffer_reports_its_pinned_staging) {
    gl::gpu::CudaPhase2BlockPool pool(16ull * 1024 * 1024);
    gl::gpu::CudaPhase2Arena arena(pool, 4);
    const gl::gpu::Phase2ProjectionCapacity projectionCapacity{
        1, 8, 256, 16384, 1024, 256, 16384, 10, 1024, 4096,
        65536, 1024, 65536, 1, 1024, 4096, 65536, 1024,
        11, 1024, 4096, 1024, 32, 4096};
    gl::gpu::CudaPhase2ProjectionBuffer projection(projectionCapacity, &arena);
    // A streamed device image never stages: construction and a streamed
    // upload leave no page-locked host memory (D-447).
    ASSERT_EQ(projection.pinnedHostBytes(), uint64_t(0));
    projection.beginStreamedUpload();
    projection.finishPhase2Upload();
    ASSERT_EQ(projection.pinnedHostBytes(), uint64_t(0));
    // The first deferred upload stages the deferred columns, a subset of the
    // device columns, through page-locked host memory; later ones reuse it.
    gl::gpu::Phase2ProjectionArena host(projectionCapacity);
    projection.upload(host);
    const uint64_t staged = projection.pinnedHostBytes();
    ASSERT_TRUE(staged > 0);
    ASSERT_TRUE(staged <= projection.fixedAllocationBytes());
    projection.upload(host);
    ASSERT_EQ(projection.pinnedHostBytes(), staged);
}

CUDA_TEST(phase2_cuda, measured_input_capacity_preserves_every_column_and_empty_reservations) {
    gl::gpu::Phase2ProjectionUsage usage{};
    usage.logicalBlocks = 1;
    usage.statements = 2;
    usage.nameRecords = 3;
    usage.nameBytes = 4;
    usage.nameSlots = 5;
    usage.ruleStringRecords = 6;
    usage.ruleStringBytes = 7;
    usage.byteMapViews = 8;
    usage.byteMapEntries = 9;
    usage.byteMapSlots = 10;
    usage.byteKeyBytes = 11;
    usage.blobRecords = 12;
    usage.blobBytes = 13;
    usage.reverseMapViews = 14;
    usage.reverseMapEntries = 15;
    usage.reverseMapSlots = 16;
    usage.reverseKeyBytes = 17;
    usage.reverseOwners = 18;
    usage.podMapViews = 19;
    usage.podMapEntries = 20;
    usage.podMapSlots = 21;
    usage.podRunValues = 22;
    usage.mandatoryStatementKeys = 23;
    usage.metadataBytes = 24;
    const auto input = gl::gpu::makePhase2BufferPlanInput(usage, {2, 4, 6, 8}, 1ull << 34, 32, 12);
    ASSERT_EQ(input.projection.logicalBlocks, 1u);
    ASSERT_EQ(input.projection.statements, 2u);
    ASSERT_EQ(input.projection.nameRecords, 3u);
    ASSERT_EQ(input.projection.nameBytes, 4u);
    ASSERT_EQ(input.projection.nameSlots, 5u);
    ASSERT_EQ(input.projection.ruleStringRecords, 6u);
    ASSERT_EQ(input.projection.ruleStringBytes, 7u);
    ASSERT_EQ(input.projection.byteMapViews, 8u);
    ASSERT_EQ(input.projection.byteMapEntries, 9u);
    ASSERT_EQ(input.projection.byteMapSlots, 10u);
    ASSERT_EQ(input.projection.byteKeyBytes, 11u);
    ASSERT_EQ(input.projection.blobRecords, 12u);
    ASSERT_EQ(input.projection.blobBytes, 13u);
    ASSERT_EQ(input.projection.reverseMapViews, 14u);
    ASSERT_EQ(input.projection.reverseMapEntries, 15u);
    ASSERT_EQ(input.projection.reverseMapSlots, 16u);
    ASSERT_EQ(input.projection.reverseKeyBytes, 17u);
    ASSERT_EQ(input.projection.reverseOwners, 18u);
    ASSERT_EQ(input.projection.podMapViews, 19u);
    ASSERT_EQ(input.projection.podMapEntries, 20u);
    ASSERT_EQ(input.projection.podMapSlots, 21u);
    ASSERT_EQ(input.projection.podRunValues, 22u);
    ASSERT_EQ(input.projection.mandatoryStatementKeys, 23u);
    ASSERT_EQ(input.projection.metadataBytes, 24u);
    ASSERT_EQ(input.tasks.tasks, 2u);
    ASSERT_EQ(input.tasks.batches, 4u);
    ASSERT_EQ(input.tasks.terms, 6u);
    ASSERT_EQ(input.tasks.stumps, 8u);
    ASSERT_EQ(input.filter.examinedRows, 1ull << 34);
    ASSERT_EQ(input.filter.retainedRows, 32u);
    ASSERT_EQ(input.filter.maximumExaminedRowsPerCall, 12u);
    const auto empty = gl::gpu::makePhase2BufferPlanInput({}, {}, 0, 0, 0);
    ASSERT_EQ(empty.projection.logicalBlocks, 1u);
    ASSERT_EQ(empty.projection.statements, 1u);
    ASSERT_EQ(empty.projection.nameRecords, 1u);
    ASSERT_EQ(empty.projection.nameBytes, 1u);
    ASSERT_EQ(empty.projection.nameSlots, 1u);
    ASSERT_EQ(empty.projection.ruleStringRecords, 1u);
    ASSERT_EQ(empty.projection.ruleStringBytes, 1u);
    ASSERT_EQ(empty.projection.byteMapViews, 1u);
    ASSERT_EQ(empty.projection.byteMapEntries, 1u);
    ASSERT_EQ(empty.projection.byteMapSlots, 1u);
    ASSERT_EQ(empty.projection.byteKeyBytes, 1u);
    ASSERT_EQ(empty.projection.blobRecords, 1u);
    ASSERT_EQ(empty.projection.blobBytes, 1u);
    ASSERT_EQ(empty.projection.reverseMapViews, 1u);
    ASSERT_EQ(empty.projection.reverseMapEntries, 1u);
    ASSERT_EQ(empty.projection.reverseMapSlots, 1u);
    ASSERT_EQ(empty.projection.reverseKeyBytes, 1u);
    ASSERT_EQ(empty.projection.reverseOwners, 1u);
    ASSERT_EQ(empty.projection.podMapViews, 1u);
    ASSERT_EQ(empty.projection.podMapEntries, 1u);
    ASSERT_EQ(empty.projection.podMapSlots, 1u);
    ASSERT_EQ(empty.projection.podRunValues, 1u);
    ASSERT_EQ(empty.projection.mandatoryStatementKeys, 1u);
    ASSERT_EQ(empty.projection.metadataBytes, 1u);
    ASSERT_TRUE(gl::gpu::CudaPhase2ProjectionBuffer::measureAllocation(empty.projection).deviceBytes > 0);
    ASSERT_EQ(gl::gpu::planPhase2Buffers(empty, 128, gl::gpu::initialPhase2WorkDemand(), {}, true).state,
        gl::gpu::Phase2BufferPlanState::fits);
}

CUDA_TEST(phase2_cuda, automatic_capacity_plans_fit_budgets_and_lend_blocks) {
    const gl::gpu::Phase2BufferPlanInput input{
        {1, 8, 256, 16384, 1024, 256, 16384, 10, 1024, 4096,
         65536, 1024, 65536, 1, 1024, 4096, 65536, 1024, 11, 1024, 4096, 1024, 32, 4096},
        {2, 8, 8, 8}, {8, 128, 64, 32}};
    const auto demand = gl::gpu::initialPhase2WorkDemand();
    constexpr uint64_t mib = 1024ull * 1024;
    for (uint64_t total : {512ull, 1024ull, 4096ull, 16384ull}) {
        const auto budget = gl::gpu::makePhase2GpuBudget(total * mib, total * mib);
        const auto plan = gl::gpu::planPhase2Buffers(input, budget.blocks, demand);
        const auto minimum = gl::gpu::planPhase2Buffers(input, budget.blocks, demand, {}, true);
        ASSERT_EQ(minimum.state, gl::gpu::Phase2BufferPlanState::fits);
        ASSERT_TRUE(minimum.totalBlocks <= plan.totalBlocks);
        ASSERT_EQ(minimum.growth.rawRequests, 1u);
        ASSERT_EQ(plan.state, gl::gpu::Phase2BufferPlanState::fits);
        ASSERT_TRUE(plan.totalBlocks <= budget.blocks);
        ASSERT_TRUE(plan.totalBlocks * 100 >= static_cast<uint64_t>(budget.blocks) * 95);
        if (total >= 4096) {
            ASSERT_TRUE(plan.growth.acceptedEvents >= (1u << 23));
            ASSERT_TRUE(plan.growth.rawRequests >= (1u << 18));
        }
        ASSERT_TRUE(plan.growth.frontierRecords <= std::max(gl::gpu::phase2FrontierExecutionRows, plan.growth.rawRequests));
        ASSERT_TRUE(plan.growth.frontierStoreRecords >= plan.growth.frontierRecords);
        ASSERT_TRUE(plan.growth.frontierStoreRecords <= std::max(plan.growth.frontierRecords,
            gl::gpu::phase2FrontierStoreMaximumRows));
        ASSERT_TRUE(minimum.growth.frontierStoreRecords >= minimum.growth.frontierRecords);
        ASSERT_EQ(plan.ordering.events, plan.growth.acceptedEvents);
        ASSERT_EQ(plan.ordering.requests, plan.growth.rawRequests);
        ASSERT_EQ(plan.evaluation.requests, plan.growth.rawRequests);
        ASSERT_EQ(plan.evaluation.reverseOwners, plan.evaluation.encodedHits);
        ASSERT_TRUE(plan.growth.prefixPayloadValues >= gl::ExecutionParameters::MAX_KEY_SLOTS);
        uint64_t sum = 0;
        for (uint32_t group = 0; group < 6; ++group) {
            ASSERT_TRUE(plan.blocks[group] * gl::gpu::kPhase2GpuBlockBytes >= plan.allocations[group].arenaBytes);
            sum += plan.blocks[group];
        }
        ASSERT_EQ(sum, plan.totalBlocks);
    }
    const auto smallBudget = gl::gpu::makePhase2GpuBudget(512 * mib, 512 * mib);
    const auto ordinary = gl::gpu::planPhase2Buffers(input, smallBudget.blocks, demand);
    auto largeInput = input;
    largeInput.projection.nameBytes = static_cast<uint32_t>(200 * mib);
    const auto large = gl::gpu::planPhase2Buffers(largeInput, smallBudget.blocks, demand);
    ASSERT_EQ(large.state, gl::gpu::Phase2BufferPlanState::fits);
    ASSERT_TRUE(large.totalBlocks <= smallBudget.blocks);
    ASSERT_TRUE(large.growth.acceptedEvents < ordinary.growth.acceptedEvents);
    largeInput.projection.nameBytes = static_cast<uint32_t>(300 * mib);
    const auto oversized = gl::gpu::planPhase2Buffers(largeInput, smallBudget.blocks, demand);
    ASSERT_EQ(oversized.state, gl::gpu::Phase2BufferPlanState::needsSmallerChunk);
    ASSERT_TRUE(oversized.totalBlocks > smallBudget.blocks);

    gl::gpu::Phase2OutputUsage outputFloor{};
    outputFloor.columns[1] = 64 * mib;
    const auto lent = gl::gpu::planPhase2Buffers(input, smallBudget.blocks, demand, outputFloor);
    ASSERT_EQ(lent.state, gl::gpu::Phase2BufferPlanState::fits);
    ASSERT_TRUE(lent.evaluation.generatedBytes >= outputFloor.columns[1]);
    ASSERT_TRUE(lent.growth.acceptedEvents < ordinary.growth.acceptedEvents);
    outputFloor.columns[1] = 1024 * mib;
    ASSERT_EQ(gl::gpu::planPhase2Buffers(input, smallBudget.blocks, demand, outputFloor).state,
        gl::gpu::Phase2BufferPlanState::needsOutputSegmentation);
    outputFloor.columns[1] = UINT64_MAX;
    ASSERT_EQ(gl::gpu::planPhase2Buffers(input, smallBudget.blocks, demand, outputFloor).state,
        gl::gpu::Phase2BufferPlanState::needsOutputSegmentation);
}

CUDA_TEST(phase2_cuda, completed_batch_feedback_redistributes_output_capacity) {
    const gl::gpu::Phase2BufferPlanInput input{
        {1, 8, 256, 16384, 1024, 256, 16384, 10, 1024, 4096,
         65536, 1024, 65536, 1, 1024, 4096, 65536, 1024, 11, 1024, 4096, 1024, 32, 4096},
        {2, 8, 8, 8}, {8, 128, 64, 32}};
    auto demand = gl::gpu::initialPhase2WorkDemand();
    const uint32_t bytes = static_cast<uint32_t>(gl::gpu::Phase2WorkDimension::bytes);
    const uint32_t requests = static_cast<uint32_t>(gl::gpu::Phase2WorkDimension::requests);
    const auto before = gl::gpu::planPhase2Buffers(input, 1024, demand);
    gl::gpu::Phase2WorkFeedback feedback{};
    for (uint32_t i = 0; i < gl::gpu::phase2WorkDimensionCount; ++i)
        feedback.occupied[i] = demand.items[i];
    feedback.drains[bytes] = 7;
    const auto oldBytes = demand.items[bytes];
    const auto oldRequests = demand.items[requests];
    gl::gpu::observePhase2WorkDemand(demand, feedback);
    ASSERT_EQ(demand.items[bytes], oldBytes * 8);
    ASSERT_EQ(demand.items[requests], oldRequests);
    const auto after = gl::gpu::planPhase2Buffers(input, 1024, demand);
    ASSERT_EQ(after.state, gl::gpu::Phase2BufferPlanState::fits);
    ASSERT_TRUE(after.evaluation.generatedBytes > before.evaluation.generatedBytes);
    feedback = {};
    const auto prior = demand.items[bytes];
    gl::gpu::observePhase2WorkDemand(demand, feedback);
    ASSERT_EQ(demand.items[bytes], prior - prior / 8);
    feedback.occupied[bytes] = UINT64_MAX;
    feedback.drains[bytes] = 1;
    gl::gpu::observePhase2WorkDemand(demand, feedback);
    ASSERT_EQ(demand.items[bytes], UINT64_MAX);
}

CUDA_TEST(phase2_cuda, minimum_output_scratch_covers_shared_cpu_span_bounds) {
    for (uint32_t blockBytes : {65536u, 262144u, 1048576u}) {
        const auto floor = gl::gpu::minimumPhase2OutputUsage(blockBytes);
        ASSERT_EQ(floor.columns[0], 1ull);
        ASSERT_EQ(floor.columns[1], 2ull * blockBytes);
        ASSERT_EQ(floor.columns[2], blockBytes / sizeof(int32_t));
        ASSERT_EQ(floor.columns[3], static_cast<uint64_t>(gl::ExecutionParameters::MAX_EXPRESSIONS + 1));
        ASSERT_EQ(floor.columns[4], 1ull);
        ASSERT_TRUE(floor.columns[5] * sizeof(gl::SealedString) <= blockBytes);
        ASSERT_TRUE((floor.columns[5] + 1) * sizeof(gl::SealedString) > blockBytes);
        ASSERT_EQ(floor.columns[6], static_cast<uint64_t>(gl::ExecutionParameters::MAX_ARITY));
    }
}
