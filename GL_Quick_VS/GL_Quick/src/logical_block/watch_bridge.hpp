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

        /// @brief One `equality1` bridge assembled across every class bucket
        ///        visible at a scope — the composed canonical text plus the
        ///        justifying equalities and their cite scopes.
        ///
        /// @details
        /// Filled by `canonicalWatchFormAt`: `text` is the input when
        /// `!changed`; each `eqText[i]` is a `(=[member,canonical])` build
        /// on the caller's out arena in applied-member order. In proof mode,
        /// `eqScope[i]` is its `findEqualityCiteScope` answer copied onto the
        /// same arena. Lookup mode preserves text and locality but leaves
        /// citation scopes empty; such a result is not a proof bridge.
        struct WatchBridge {
            static constexpr int32_t CAP = 4 * ExecutionParameters::MAX_ARITY;
            StrSpan text{};
            bool changed = false;
            bool localEquality = false;
            int32_t eqN = 0;
            StrSpan eqText[CAP];
            StrSpan eqScope[CAP];
        };
}
