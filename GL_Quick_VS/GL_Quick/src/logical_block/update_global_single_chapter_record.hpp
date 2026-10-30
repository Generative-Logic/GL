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
#include "../memory_infra/temporary_witness_store.hpp"
namespace gl {
namespace tests { struct RuntimeSupportFixture; }

        /// @brief One proven single-chapter (non-auxy) theorem staged for the post-join
        ///        `updateGlobalSingleChapter` broadcast — the sealed-bytes replacement
        ///        for the former `std::vector<std::tuple<std::string,int>>`
        ///        `updateGlobalSingleChapterTuples`.
        ///
        /// @details
        /// A trivially-copyable POD: `theorem` is a `SealedString` on the shared
        /// `updateGlobalSingleChapterPages` set — the proven theorem's NameMap-decoded
        /// bytes must cross the parallel → single-threaded join VERBATIM (the LB
        /// deloads after its burst, so a deferred decode is impossible). `coreId`
        /// carries through so the drain replay is byte-identical to the former
        /// tuple order. The record rides the page set's own record chain
        /// (`appendRecord`, I-135).
        struct UpdateGlobalSingleChapterRec {
            SealedString theorem;
            // The sealing worker's dispatch slot — forwarded to
            // `updateGlobalSingleChapter` for the compaction queue. A scheduling
            // race outcome: it must never participate in the drain order
            // (see `updateGlobalSingleChapterLess`).
            int coreId;
            // The sealing LB — address-stable for the LB's whole life
            // (I-109), read at the single-threaded drain by the
            // vacuous-premise emission gate (`hasContradictedAncestor`).
            Memory* producer;
            // Registration verdict, sealed at discharge time from the
            // producer's own resident level rows (a drain-time re-probe
            // could miss: end-of-burst sanitize rewrites statement text
            // and the LB may be cold post-join). The drain runs the goal
            // lifecycle unconditionally and refuses ONLY registration
            // when false.
            bool allLevelsInvolved;
        };
}
