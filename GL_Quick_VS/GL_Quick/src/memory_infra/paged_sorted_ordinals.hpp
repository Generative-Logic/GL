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

#include "paged_vector.hpp"

namespace gl {

    /// @brief Sort an ordinal stream with paged runs and bounded merge fan-in.
    ///
    /// @details
    /// Initial runs use one bounded contiguous scratch buffer. Their ordinals
    /// and run descriptors live in page-tier vectors. Pairwise merge passes use
    /// two source cursors, so neither a run-count array nor a contiguous index
    /// imposes a processing ceiling. Comparators observe logical ordinals only.
    ///
    /// @tparam Less Strict-weak comparator over zero-based source ordinals.
    /// @invariant The output is a permutation sorted by the supplied comparator.
    template <typename Less>
    class PagedSortedOrdinals {
        struct Run {
            int32_t begin;
            int32_t count;
        };

        DirtyState dirty_{ DirtyState::Clean };
        PagedVector<int32_t> first_;
        PagedVector<int32_t> second_;
        PagedVector<Run> firstRuns_;
        PagedVector<Run> secondRuns_;
        bool firstIsOutput_{ true };
        int32_t cursor_{ 0 };

    public:
        /// @brief Construct and merge sorted runs on a caller-owned arena.
        ///
        /// @details
        /// Page storage survives the constructor's temporary byte-bump rewind.
        /// The caller keeps the arena resident and alive until this index dies.
        /// Equal comparator values retain left-run order during merging.
        ///
        /// @param arena Resident page source and temporary run-sort scratch.
        /// @param total Number of source ordinals, including zero for no work.
        /// @param less Comparator that remains valid throughout construction.
        /// @param runSlots Maximum ordinals in the bounded initial sort buffer.
        /// @return A sorted index ready for next().
        /// @invariant Every source ordinal appears exactly once.
        PagedSortedOrdinals(LbArena& arena, int32_t total, Less less,
            int32_t runSlots)
            : first_(&arena, &dirty_), second_(&arena, &dirty_),
              firstRuns_(&arena, &dirty_), secondRuns_(&arena, &dirty_) {
            assert(total >= 0 && runSlots > 0);
            if (total == 0) return;
            const int32_t bufferCount = std::min(total, runSlots);
            assert(bufferCount <= INT32_MAX / static_cast<int32_t>(sizeof(int32_t)));
            const ArenaOffset mark = arena.cursor();
            const ArenaOffset offset = arena.alloc(
                bufferCount * static_cast<int32_t>(sizeof(int32_t)),
                static_cast<int32_t>(alignof(int32_t)));
            int32_t* const buffer = reinterpret_cast<int32_t*>(arena.resolve(offset));
            for (int32_t begin = 0; begin < total;) {
                const int32_t count = std::min(bufferCount, total - begin);
                for (int32_t index = 0; index < count; ++index)
                    buffer[index] = begin + index;
                std::sort(buffer, buffer + count, less);
                firstRuns_.push_back(Run{begin, count});
                for (int32_t index = 0; index < count; ++index)
                    first_.push_back(buffer[index]);
                begin += count;
            }
            arena.popTo(mark);

            while ((firstIsOutput_ ? firstRuns_ : secondRuns_).size() > 1) {
                auto& input = firstIsOutput_ ? first_ : second_;
                auto& output = firstIsOutput_ ? second_ : first_;
                auto& runs = firstIsOutput_ ? firstRuns_ : secondRuns_;
                auto& outputRuns = firstIsOutput_ ? secondRuns_ : firstRuns_;
                output.clear();
                outputRuns.clear();
                for (int32_t index = 0; index < runs.size(); index += 2) {
                    const Run left = runs[index];
                    const Run right = index + 1 < runs.size()
                        ? runs[index + 1] : Run{left.begin + left.count, 0};
                    int32_t leftAt = left.begin;
                    int32_t rightAt = right.begin;
                    const int32_t leftEnd = left.begin + left.count;
                    const int32_t rightEnd = right.begin + right.count;
                    outputRuns.push_back(Run{output.size(), left.count + right.count});
                    while (leftAt < leftEnd || rightAt < rightEnd) {
                        const bool takeLeft = rightAt == rightEnd
                            || (leftAt < leftEnd
                                && !less(input[rightAt], input[leftAt]));
                        output.push_back(input[takeLeft ? leftAt++ : rightAt++]);
                    }
                }
                assert(output.size() == total);
                input.clear();
                runs.clear();
                firstIsOutput_ = !firstIsOutput_;
            }
        }

        /// @brief Consume the next ordinal in canonical sorted order.
        ///
        /// @details
        /// No comparison or allocation occurs while consuming the sorted index.
        /// Calling after the known total is a contract violation.
        ///
        /// @return Next source ordinal.
        /// @invariant The cursor advances by exactly one and stays in range.
        int32_t next() {
            const auto& output = firstIsOutput_ ? first_ : second_;
            assert(cursor_ < output.size());
            return output[cursor_++];
        }

        /// @brief Rewind consumption while retaining the sorted index.
        ///
        /// @details
        /// Replaying does not repeat sorting or alter page ownership.
        ///
        /// @return Nothing.
        /// @invariant The next call starts at the same first sorted ordinal.
        void reset() {
            cursor_ = 0;
        }
    };

}
