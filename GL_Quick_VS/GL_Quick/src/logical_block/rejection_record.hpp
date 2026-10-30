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

        /// @brief One buffered disintegration-rejection record (interned ids).
        ///
        /// @details
        /// Flat POD over @ref RejectionStore's string interner and sub-arrays:
        /// `varId` (the witness var — the drain's admitted gate), `s1Id` / `s2Id`
        /// (removedU/concreteConstituent, markedExpr), `s3Id` (compactExpr; 0 for the
        /// integration side, which has none), and `[sibStart, sibStart+sibCount)` /
        /// `[levStart, levStart+levCount)` ranges into the sibling-id and level columns.
        struct RejRec {
            int32_t varId, s1Id, s2Id, s3Id;
            int32_t sibStart, sibCount;
            int32_t levStart, levCount;
        };
}
