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
#include "or_implication_refuter.hpp"
namespace gl {
namespace tests { struct RuntimeSupportFixture; }

        /// @brief Bounded selection passed directly to firing materialization.
        /// @details Refuters, source metadata and scope decisions retain the
        /// existing deterministic choices. The record is valid only for immediate
        /// consumption before another operation mutates the block. Ordinary
        /// hypothetical-scope gates remain part of materialization.
        struct OrImplicationSelection {
            int64_t sourceKey;
            NameId lookupScope;
            OrImplicationSource source;
            NameId canonicalSource;
            NameId sourceProofScope;
            NameId premiseScope;
            NameId resultScope;
            OrImplicationRefuter choices[ExecutionParameters::kMaxReducedOrLeaves];
            bool used[ExecutionParameters::kMaxReducedOrLeaves];
            NameId factProofScopes[ExecutionParameters::kMaxReducedOrLeaves];
            int32_t refuted;
            int32_t survivors;
            int32_t headLeaf;
            bool localPremise;
        };
}
