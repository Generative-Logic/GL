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
#include "../memory.hpp"
namespace gl {
/// @brief Sorted-unique pull over the ordinals `0..total-1` of one
        ///        staged key's items: the block-bounded sort
        ///        (`ChunkSortedOrdinals`) with the duplicates skipped on the
        ///        way out — the flushes' merge source.
        ///
        /// @details
        /// `next` hands out the ordinals in ascending @p Less order, once per
        /// distinct element (@p Equal); `reset` replays the same sequence (a
        /// counting pass before a writing pass). The chunk arrays ride the
        /// caller's arena byte-bump tier under the caller's mark / pop window.
        ///
        /// @tparam Less  Strict weak order over ordinals, `bool(int32_t, int32_t)`.
        /// @tparam Equal Equivalence over ordinals, `bool(int32_t, int32_t)`.
        template <typename Less, typename Equal>
        class StagedSortedUnique {
        public:
            /// @brief Build the sort over `total` ordinals on @p arena.
            /// @details Borrow the arena for a replayable sorted-unique traversal.
            /// @param arena Scratch storage retained throughout this iterator's use.
            /// @param total Number of source ordinals.
            /// @param less Strict ordering of source ordinals.
            /// @param equal Equivalence predicate over source ordinals.
            /// @return Constructed iterator.
            StagedSortedUnique(ScratchArena& arena, int32_t total, Less less, Equal equal)
                : sorted_(arena, total, 0, less), total_(total), equal_(equal) {}

            /// @brief The next distinct ordinal in sorted order.
            /// @details Skip equivalent consecutive entries; exhaustion is a defined result.
            /// @param out Receives the ordinal.
            /// @return `false` when exhausted.
            bool next(int32_t& out) {
                while (consumed_ < total_) {
                    const int32_t o = sorted_.next();
                    ++consumed_;
                    if (havePrev_ && equal_(prev_, o)) continue;
                    prev_ = o;
                    havePrev_ = true;
                    out = o;
                    return true;
                }
                return false;
            }

            /// @brief Replay the same sequence from the start.
            /// @details Preserve the sorted storage and reset only traversal state.
            /// @return Nothing.
            void reset() {
                sorted_.reset();
                consumed_ = 0;
                havePrev_ = false;
            }

        private:
            ChunkSortedOrdinals<Less> sorted_;
            int32_t total_;
            Equal equal_;
            int32_t consumed_ = 0;
            bool havePrev_ = false;
            int32_t prev_ = 0;
        };
} // namespace gl
