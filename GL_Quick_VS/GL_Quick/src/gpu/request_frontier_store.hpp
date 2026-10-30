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

#include "phase2_projection.hpp"
#include "../memory_infra/extent_file.hpp"
#include "../memory_infra/paged_vector.hpp"
#include <algorithm>
#include <cassert>
#include <cstdint>

namespace gl::gpu {

    /// @brief Normal maximum row payload of one resident request frontier.
    inline constexpr int32_t phase2FrontierHostCacheRows =
        (32 * 1024 * 1024) / sizeof(DeviceGrowthNode);

    /// @brief Borrowed, prepared files and resident row budget for one growth call.
    /// @details Null files select the existing resident-only policy. Both files
    /// must be distinct, open, and warmed on the growth thread before use.
    struct RequestFrontierStorageOptions {
        PositionedFile* files[2]{nullptr, nullptr};
        int32_t cacheRows{phase2FrontierHostCacheRows};
    };

    /// @brief Disk traffic and retained resident storage for one frontier.
    struct RequestFrontierStorageStats {
        uint64_t writes{0}, reads{0}, writtenBytes{0}, readBytes{0};
        uint64_t peakResidentPayloadBytes{0}, peakResidentStorageBytes{0};
    };

    /// @brief An ordered request frontier with a disk prefix and bounded tail.
    /// @details Disk-backed stores flush a full tail before growing it further.
    /// The existing paged storage supplies resident-only mode and the bounded
    /// tail. File offsets are process-local scratch, never proof identities.
    /// @invariant Disk prefix followed by the resident tail equals append order.
    class RequestFrontierStore {
    public:
        /// @brief Bind an empty frontier to its arena and optional prepared file.
        /// @details The caller owns the arena and file through this store's life.
        /// No file setup or heap allocation occurs here.
        /// @param arena Resident page owner.
        /// @param file Prepared file, or null for resident-only operation.
        /// @param cacheRows Positive resident payload limit when disk is enabled.
        /// @return An empty frontier.
        RequestFrontierStore(LbArena& arena, PositionedFile* file, int32_t cacheRows)
            : resident_(&arena, &dirty_), file_(file), cacheRows_(cacheRows) {
            assert(cacheRows_ > 0);
            assert(file_ == nullptr || file_->isOpen());
            if (file_ != nullptr) {
                const int32_t bytes = static_cast<int32_t>(arena.blockBytes());
                transferBytes_ = bytes;
                stats_.peakResidentStorageBytes = static_cast<uint64_t>(bytes);
                transferRows_ = bytes / sizeof(DeviceGrowthNode);
                assert(transferRows_ > 0);
                transfer_ = reinterpret_cast<DeviceGrowthNode*>(arena.resolve(arena.alloc(bytes, 16)));
            }
        }

        RequestFrontierStore(const RequestFrontierStore&) = delete;
        RequestFrontierStore& operator=(const RequestFrontierStore&) = delete;

        /// @brief Count the complete logical frontier.
        /// @details Resident tail and disk prefix are disjoint ordered ranges.
        /// @return Number of rows in both ranges.
        uint64_t size() const {
            return diskRows_ + static_cast<uint64_t>(resident_.size());
        }

        /// @brief Test whether the logical frontier has no rows.
        /// @details Retained storage does not count as logical content.
        /// @return True exactly when size is zero.
        bool empty() const { return size() == 0; }

        /// @brief Read cumulative traffic and resident high-water counters.
        /// @details Counters survive logical resets within this growth call.
        /// @return Store-local observation counters.
        const RequestFrontierStorageStats& stats() const { return stats_; }

        /// @brief Append rows in their original order.
        /// @details A full disk-backed tail is written before another row is
        /// retained. Large incoming runs are split without changing identities.
        /// @param rows Caller-owned source, non-null for a nonempty run.
        /// @param count Number of rows, nonnegative.
        /// @return Nothing.
        void appendRun(const DeviceGrowthNode* rows, int32_t count) {
            assert(count >= 0 && (count == 0 || rows != nullptr));
            assert(size() <= static_cast<uint64_t>(INT64_MAX) / sizeof(DeviceGrowthNode)
                - static_cast<uint64_t>(count));
            int32_t done = 0;
            while (done < count) {
                if (file_ != nullptr && resident_.size() == cacheRows_) flushTail();
                const int32_t available = file_ != nullptr
                    ? cacheRows_ - resident_.size() : INT32_MAX - resident_.size();
                assert(available > 0);
                const int32_t copied = std::min(available, count - done);
                resident_.appendRun(rows + done, copied);
                done += copied;
                stats_.peakResidentPayloadBytes = std::max(stats_.peakResidentPayloadBytes,
                    static_cast<uint64_t>(resident_.size()) * sizeof(DeviceGrowthNode));
                stats_.peakResidentStorageBytes = std::max(stats_.peakResidentStorageBytes,
                    static_cast<uint64_t>(resident_.liveBytes()) + transferBytes_);
            }
        }

        /// @brief Copy a bounded logical window into caller-owned transfer memory.
        /// @details A window can cross the disk/tail boundary. Every disk read
        /// addresses rows already fully written by this store.
        /// @param at First logical row.
        /// @param count Number of rows, nonnegative.
        /// @param destination Writable output for count rows.
        /// @return Nothing.
        void copyRunTo(uint64_t at, int32_t count, DeviceGrowthNode* destination) {
            assert(count >= 0 && (count == 0 || destination != nullptr));
            assert(at <= size() && static_cast<uint64_t>(count) <= size() - at);
            int32_t copied = 0;
            if (at < diskRows_ && count > 0) {
                copied = static_cast<int32_t>(std::min<uint64_t>(count, diskRows_ - at));
                const uint64_t bytes = static_cast<uint64_t>(copied) * sizeof(DeviceGrowthNode);
                file_->readAt(static_cast<int64_t>(at * sizeof(DeviceGrowthNode)),
                    destination, static_cast<int64_t>(bytes));
                ++stats_.reads;
                stats_.readBytes += bytes;
            }
            if (copied < count) {
                const uint64_t tailAt = at + copied - diskRows_;
                assert(tailAt <= INT32_MAX);
                resident_.copyRunTo(static_cast<int32_t>(tailAt), count - copied,
                    destination + copied);
            }
        }

        /// @brief Remove stopped calls while preserving surviving row order.
        /// @details Resident-only content compacts in place. A disk-backed
        /// frontier reads bounded windows and writes survivors behind its read
        /// cursor, so no unread row is overwritten. The final tail moves to disk.
        /// @param keepByCall One zero-or-one retention flag per original call.
        /// @param callCount Number of flags.
        /// @param scratch Caller-owned row buffer, disjoint from this store.
        /// @param scratchRows Positive scratch capacity.
        /// @return Number of removed rows.
        uint64_t pruneByCall(const uint8_t* keepByCall, uint32_t callCount,
            DeviceGrowthNode* scratch, int32_t scratchRows) {
            assert(keepByCall != nullptr && callCount > 0);
            assert(scratch != nullptr && scratchRows > 0);
            const uint64_t original = size();
            uint64_t kept = 0;
            if (diskRows_ == 0) {
                const int32_t count = resident_.size();
                for (int32_t at = 0; at < count; ++at) {
                    const DeviceGrowthNode row = resident_[at];
                    assert(row.callIndex < callCount && keepByCall[row.callIndex] <= 1);
                    if (keepByCall[row.callIndex] != 0) {
                        if (kept != static_cast<uint64_t>(at))
                            resident_.setAt(static_cast<int32_t>(kept), row);
                        ++kept;
                    }
                }
                if (kept != original) resident_.truncate(static_cast<int32_t>(kept));
            } else {
                for (uint64_t at = 0; at < original;) {
                    const int32_t count = static_cast<int32_t>(
                        std::min<uint64_t>(scratchRows, original - at));
                    copyRunTo(at, count, scratch);
                    int32_t selected = 0;
                    for (int32_t i = 0; i < count; ++i) {
                        assert(scratch[i].callIndex < callCount
                            && keepByCall[scratch[i].callIndex] <= 1);
                        if (keepByCall[scratch[i].callIndex] != 0)
                            scratch[selected++] = scratch[i];
                    }
                    if (selected > 0) writeRows(kept, scratch, selected);
                    kept += selected;
                    at += count;
                }
                resident_.resetKeepingPages();
                diskRows_ = kept;
            }
            return original - kept;
        }

        /// @brief Empty logical content while retaining bounded reusable pages.
        /// @details Old disk bytes become inaccessible and future writes start
        /// at zero. No file grows merely because the store is reused.
        /// @return Nothing.
        void resetKeepingPages() {
            diskRows_ = 0;
            resident_.resetKeepingPages();
        }

    private:
        /// @brief Write a checked row range into this store's prepared file.
        /// @details File growth uses the existing coarse extent growth unit.
        /// Transfer errors assert in PositionedFile; there is no alternate path.
        /// @param at First output row.
        /// @param rows Source row buffer.
        /// @param count Positive row count.
        /// @return Nothing.
        void writeRows(uint64_t at, const DeviceGrowthNode* rows, int32_t count) {
            assert(file_ != nullptr && rows != nullptr && count > 0);
            constexpr uint64_t maxRows = static_cast<uint64_t>(INT64_MAX)
                / sizeof(DeviceGrowthNode);
            assert(at <= maxRows && static_cast<uint64_t>(count) <= maxRows - at);
            const uint64_t end = (at + count) * sizeof(DeviceGrowthNode);
            if (end > static_cast<uint64_t>(file_->fileSize())) {
                constexpr uint64_t chunk = 256ull << 20;
                assert(end <= static_cast<uint64_t>(INT64_MAX) - (chunk - 1));
                file_->grow(static_cast<int64_t>((end + chunk - 1) / chunk * chunk));
            }
            const uint64_t bytes = static_cast<uint64_t>(count) * sizeof(DeviceGrowthNode);
            file_->writeAt(static_cast<int64_t>(at * sizeof(DeviceGrowthNode)),
                rows, static_cast<int64_t>(bytes));
            ++stats_.writes;
            stats_.writtenBytes += bytes;
        }

        /// @brief Move the full resident tail behind the existing disk prefix.
        /// @details Bounded spans are written synchronously before
        /// its pages are reused. Metadata remains constant in total row count.
        /// @return Nothing.
        void flushTail() {
            assert(file_ != nullptr && resident_.size() == cacheRows_);
            int32_t at = 0;
            while (at < resident_.size()) {
                int32_t count = 0;
                count = std::min(transferRows_, resident_.size() - at);
                resident_.copyRunTo(at, count, transfer_);
                writeRows(diskRows_ + at, transfer_, count);
                at += count;
            }
            diskRows_ += resident_.size();
            resident_.resetKeepingPages();
        }

        DirtyState dirty_{DirtyState::Clean};
        PagedVector<DeviceGrowthNode> resident_;
        PositionedFile* file_;
        int32_t cacheRows_;
        uint64_t diskRows_{0};
        RequestFrontierStorageStats stats_{};
        DeviceGrowthNode* transfer_{nullptr};
        int32_t transferRows_{0};
        uint64_t transferBytes_{0};
    };
}
