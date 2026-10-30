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

        /// @brief One record in @ref CollectedArena — a statement key plus one
        ///        derived item (an implication, a child statement, or nothing).
        ///
        /// @details
        /// `kind` selects the payload: 0 = implication (`a` is the original
        /// expression, `b` is the validity name), 1 = child statement (`a` is
        /// the child bytes, `b` empty), 2 = empty-key marker (both empty — the
        /// load-bearing `collected[K];` "ensure entry exists" that forceDeep /
        /// Pass-A iterate), 3 = deferred OR (`a` is its raw expression, `b`
        /// is its enclosing witness). All strings live in the owning arena.
        struct CollectedRec {
            int32_t keyId;   ///< interner id of the statement key
            int32_t aId;     ///< kind 0: impl original; kind 1: child; else 0
            int32_t bId;     ///< kind 0: impl validity; else 0
            uint8_t kind;    ///< 0 implication, 1 child, 2 empty key, 3 deferred OR
        };
}
