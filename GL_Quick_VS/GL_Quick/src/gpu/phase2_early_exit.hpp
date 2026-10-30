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

#include "phase2_cuda.hpp"
#include "../memory.hpp"
#include "../memory_infra/paged_sorted_ordinals.hpp"

namespace gl::gpu {

    /// @brief Original-order cutoff selection across bounded GPU downloads.
    /// @details All firing content and deactivation verdicts arrive from CUDA.
    /// Static host pages preserve ordering metadata and pending ownership until
    /// the complete original request prefix is known. A sole final drain uses
    /// its device positions directly, without an accepted-event host ledger.
    class Phase2EarlyExitLedger {
        struct Rank { uint64_t position{}; uint32_t order{}; };
        struct Pending {
            FiringRecord record;
            uint64_t position{};
            uint32_t request{}, block{}, part{};
            bool trigger{};
        };
        struct Cut { uint64_t position{UINT64_MAX}; uint32_t part{UINT32_MAX}, request{UINT32_MAX}; };
        const Phase2ProjectionArena& projection_;
        const Phase2TaskProjectionArena& tasks_;
        const Phase2GrowthScheduleArena& schedule_;
        LbArena arena_{&gpuHostStorage()};
        DirtyState dirty_{DirtyState::Clean};
        PagedVector<DeviceAcceptedGrowthEvent> events_{&arena_, &dirty_};
        PagedVector<uint32_t> requests_{&arena_, &dirty_};
        PagedVector<Pending> pending_{&arena_, &dirty_};
        PagedVector<Cut> drainCuts_{&arena_, &dirty_};
        uint32_t eventBase_{}, requestBase_{}, drainEventCount_{}, drainRequestCount_{};
        uint32_t drains_{};
        bool global_{};

        /// @brief Compare only the original per-call request identity.
        /// @details Premise identifiers are equality keys, never lexical ranks.
        /// @param left First retained request ordinal.
        /// @param right Second retained request ordinal.
        /// @return Negative, zero or positive for the exact identity comparison.
        int compareRequests(uint32_t left, uint32_t right) const {
            const auto& a = events_[static_cast<int32_t>(requests_[static_cast<int32_t>(left)])];
            const auto& b = events_[static_cast<int32_t>(requests_[static_cast<int32_t>(right)])];
            if (a.callIndex != b.callIndex) return a.callIndex < b.callIndex ? -1 : 1;
            if (a.count != b.count) return a.count < b.count ? -1 : 1;
            const auto& task = tasks_.tasks[schedule_.calls[a.callIndex].taskIndex];
            const auto& block = projection_.logicalBlocks[task.logicalBlockIndex];
            for (NameId i = 0; i < a.count; ++i) {
                const auto& x = projection_.statements[block.statementOffset + a.statementIndices[i]];
                const auto& y = projection_.statements[block.statementOffset + b.statementIndices[i]];
                if (x.originalId != y.originalId) return x.originalId < y.originalId ? -1 : 1;
                if (x.validityId != y.validityId) return x.validityId < y.validityId ? -1 : 1;
            }
            return 0;
        }

    public:
        uint64_t evaluatedRequests{};

        /// @brief Bind one immutable logical projection and its call schedule.
        /// @details The three images remain alive until selection completes.
        /// @param projection Host twin of the frozen device projection.
        /// @param tasks Original task and part descriptors.
        /// @param schedule Original generator-call sequence.
        /// @return An empty cutoff ledger with no host-pool allocations.
        Phase2EarlyExitLedger(const Phase2ProjectionArena& projection,
            const Phase2TaskProjectionArena& tasks, const Phase2GrowthScheduleArena& schedule)
            : projection_(projection), tasks_(tasks), schedule_(schedule) {
            arena_.setPurpose(PoolPurpose::GpuHost);  // pool attribution
        }

        /// @brief Start metadata for one frozen request drain.
        /// @details An interruption enables the global ledger before any event
        /// slots are reused. A sole final drain already has global positions.
        /// @param finalDrain True only at complete generation exhaustion.
        /// @param eventCount Accepted events in the current device buffer.
        /// @param requestCount Locally deduplicated request tokens in this drain.
        /// @return Whether the caller must retain events and request tokens.
        bool beginDrain(bool finalDrain, uint32_t eventCount, uint32_t requestCount) {
            assert(drains_ == 0 || global_);
            global_ = global_ || !finalDrain || drains_ != 0;
            ++drains_;
            eventBase_ = static_cast<uint32_t>(events_.size());
            requestBase_ = static_cast<uint32_t>(requests_.size());
            drainEventCount_ = eventCount;
            drainRequestCount_ = requestCount;
            evaluatedRequests += requestCount;
            if (drainCuts_.empty())
                for (uint32_t block = 0; block < projection_.logicalBlocks.size(); ++block)
                    drainCuts_.push_back(Cut{});
            else
                for (int32_t block = 0; block < drainCuts_.size(); ++block)
                    drainCuts_.setAt(block, Cut{});
            return global_;
        }

        /// @brief Collect known cutoff metadata before sealing a download.
        ///
        /// @details Within one request drain, device growth positions and request
        /// indices preserve their original relative order. A trigger excludes
        /// later requests of its own part even if a future download changes the
        /// final winning part. Other parts remain available for final selection.
        ///
        /// @param rows Downloaded headers with CUDA deactivation verdicts.
        /// @param count Number of headers in the download.
        /// @return Nothing.
        void observeHeaders(const DevicePhase2FiringRecord* rows, uint32_t count) {
            assert(drains_ > 0 && (count == 0 || rows != nullptr));
            for (uint32_t i = 0; i < count; ++i) {
                const auto& record = rows[i];
                assert(record.logicalBlockIndex < static_cast<uint32_t>(drainCuts_.size()));
                assert(record.requestIndex < drainRequestCount_);
                if ((record.flags & deviceFiringDoomTrigger) == 0) continue;
                const auto& cut = drainCuts_[record.logicalBlockIndex];
                if (record.growthPosition < cut.position || (record.growthPosition == cut.position
                    && (record.partOrdinal < cut.part || (record.partOrdinal == cut.part
                        && record.requestIndex < cut.request))))
                    drainCuts_.setAt(record.logicalBlockIndex,
                        Cut{record.growthPosition, record.partOrdinal, record.requestIndex});
            }
        }

        /// @brief Decide whether a header still needs host-owned firing content.
        ///
        /// @details A known trigger can only remove later requests of that same
        /// part. Every output of its own request and every competing part remain
        /// available, including when a later drain reveals an earlier trigger.
        ///
        /// @param record Header from the current request drain.
        /// @return True when final cutoff selection may retain this record.
        bool retainsHeader(const DevicePhase2FiringRecord& record) const {
            assert(record.logicalBlockIndex < static_cast<uint32_t>(drainCuts_.size()));
            const auto& cut = drainCuts_[record.logicalBlockIndex];
            return record.partOrdinal != cut.part || record.growthPosition < cut.position
                || (record.growthPosition == cut.position && record.requestIndex <= cut.request);
        }

        /// @brief Retain consecutive accepted-event metadata before reuse.
        /// @details Physical append order remains separate from processor order.
        /// @param rows Downloaded event rows in device-index order.
        /// @param count Number of consecutive rows supplied.
        /// @return Nothing.
        void appendEvents(const DeviceAcceptedGrowthEvent* rows, uint32_t count) {
            assert(global_ && static_cast<uint64_t>(events_.size()) + count <= eventBase_ + drainEventCount_);
            for (uint32_t i = 0; i < count; ++i) events_.push_back(rows[i]);
        }

        /// @brief Retain request-to-event identities across device windows.
        /// @details Token order remains the exact local ordered-request index.
        /// @param rows Consecutive ordered request tokens.
        /// @param count Number of supplied tokens.
        /// @return Nothing.
        void appendRequests(const DeviceOrderedRequestToken* rows, uint32_t count) {
            assert(global_ && static_cast<uint64_t>(requests_.size()) + count <= requestBase_ + drainRequestCount_);
            for (uint32_t i = 0; i < count; ++i) {
                assert(rows[i].eventIndex < drainEventCount_);
                requests_.push_back(eventBase_ + rows[i].eventIndex);
            }
        }

        /// @brief Stage complete CUDA-produced firing content until selection.
        /// @details Payload spans already belong to their final logical-block
        /// output pages; this record copy neither evaluates nor reconstructs it.
        /// @param source Original device header including its trigger verdict.
        /// @param record Complete sealed content, including assembled markers.
        /// @return Nothing.
        void receive(const DevicePhase2FiringRecord& source, const FiringRecord& record) {
            assert(source.requestIndex < drainRequestCount_);
            pending_.push_back(Pending{record, source.growthPosition,
                requestBase_ + source.requestIndex, source.logicalBlockIndex,
                source.partOrdinal, (source.flags & deviceFiringDoomTrigger) != 0});
        }

        /// @brief Select the original inclusive winning-request prefix.
        /// @details Global metadata recovers processor growth positions and
        /// earliest duplicate ownership after all drains. Each block retains
        /// only its winning part through every output of the triggering request.
        /// @param outputs Filling pages already owning every staged payload.
        /// @param outputCount Number of logical-block output pages.
        /// @param doomLines Output original packed growth-position/part cutoffs.
        /// @return Number of retained firing records appended to output chains.
        /// @invariant Buffer size and physical download order cannot select output.
        uint64_t finish(SealedPageSet* const* outputs, uint32_t outputCount, int64_t* doomLines) {
            assert(outputCount == projection_.logicalBlocks.size() && doomLines != nullptr);
            PagedVector<Rank> ranks(&arena_, &dirty_);
            PagedVector<uint8_t> requestKeep(&arena_, &dirty_);
            PagedVector<Cut> cuts(&arena_, &dirty_);
            for (uint32_t block = 0; block < outputCount; ++block) cuts.push_back(Cut{});
            if (global_) {
                assert(events_.size() == eventBase_ + drainEventCount_);
                assert(requests_.size() == requestBase_ + drainRequestCount_);
                for (int32_t i = 0; i < events_.size(); ++i) ranks.push_back(Rank{});
                const auto eventLess = [&](int32_t left, int32_t right) {
                    const auto& a = events_[left]; const auto& b = events_[right];
                    const auto ta = schedule_.calls[a.callIndex].taskIndex;
                    const auto tb = schedule_.calls[b.callIndex].taskIndex;
                    if (ta != tb) return ta < tb;
                    if (a.callIndex != b.callIndex) return a.callIndex < b.callIndex;
                    if (a.runOrdinal != b.runOrdinal) return a.runOrdinal < b.runOrdinal;
                    for (int i = ExecutionParameters::MAX_EXPRESSIONS - 1; i >= 0; --i) {
                        const auto x = growthEventPathComponent(a, i), y = growthEventPathComponent(b, i);
                        if (x != y) return x < y;
                    }
                    return left < right;
                };
                PagedSortedOrdinals sorted(arena_, events_.size(), eventLess,
                    ExecutionParameters::kFiringRecordSortChunk);
                uint32_t previousTask = UINT32_MAX;
                uint64_t position = 0;
                for (int32_t order = 0; order < events_.size(); ++order) {
                    const auto index = sorted.next(); const auto& event = events_[index];
                    const auto task = schedule_.calls[event.callIndex].taskIndex;
                    if (task != previousTask) { previousTask = task; position = 0; }
                    if ((event.flags & kDeviceGrowthEventSubkeySatisfied) != 0) ++position;
                    ranks.setAt(index, Rank{position, static_cast<uint32_t>(order)});
                }
                for (int32_t i = 0; i < requests_.size(); ++i) requestKeep.push_back(1);
                PagedSortedOrdinals requestOrder(arena_, requests_.size(), [&](int32_t a, int32_t b) {
                    const int key = compareRequests(a, b);
                    return key != 0 ? key < 0 : ranks[requests_[a]].order < ranks[requests_[b]].order;
                }, ExecutionParameters::kFiringRecordSortChunk);
                int32_t previous = -1;
                evaluatedRequests = 0;
                for (int32_t i = 0; i < requests_.size(); ++i) {
                    const auto request = requestOrder.next();
                    if (previous >= 0 && compareRequests(previous, request) == 0)
                        requestKeep.setAt(request, 0);
                    else { previous = request; ++evaluatedRequests; }
                }
            }
            const auto rankOf = [&](const Pending& record) {
                return global_ ? ranks[requests_[static_cast<int32_t>(record.request)]]
                    : Rank{record.position, record.request};
            };
            for (int32_t i = 0; i < pending_.size(); ++i) {
                const auto& record = pending_[i];
                if (global_ && requestKeep[static_cast<int32_t>(record.request)] == 0) continue;
                if (!record.trigger) continue;
                const auto rank = rankOf(record); const auto& current = cuts[record.block];
                if (rank.position < current.position || (rank.position == current.position
                    && (record.part < current.part || (record.part == current.part && rank.order < current.request))))
                    cuts.setAt(record.block, Cut{rank.position, record.part, rank.order});
            }
            for (uint32_t block = 0; block < outputCount; ++block) {
                const auto& cut = cuts[block];
                doomLines[block] = INT64_MAX;
                if (cut.position != UINT64_MAX) {
                    assert(cut.position < (uint64_t{1} << 48) && cut.part < (uint32_t{1} << 14));
                    doomLines[block] = static_cast<int64_t>((cut.position << 14) | cut.part);
                }
            }
            uint64_t retained = 0;
            for (int32_t i = 0; i < pending_.size(); ++i) {
                const auto& record = pending_[i];
                if (global_ && requestKeep[static_cast<int32_t>(record.request)] == 0) continue;
                const auto rank = rankOf(record); const auto& cut = cuts[record.block];
                if (cut.position != UINT64_MAX && (record.part != cut.part
                    || rank.position > cut.position || (rank.position == cut.position && rank.order > cut.request))) continue;
                outputs[record.block]->appendRecord(record.record);
                ++retained;
            }
            return retained;
        }
    };
}
