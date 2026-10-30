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
#include "../memory_infra/scratch_arena.hpp"
namespace gl {
    /// @brief One emitted `multiplyImplication` copy as an arena reference — a
    ///        virtual byte-bump offset into the caller's string-tier
    ///        `outStrArena` plus its length.
    ///
    /// @details
    /// The span-form `multiplyImplication` emits each surviving multiplied copy
    /// as a `CopyRef` into a caller-owned `PagedVector<CopyRef>`; the caller
    /// resolves `outStrArena.resolve(off)` to read the bytes. A virtual
    /// `ArenaOffset` (never a raw `char*`) so the reference is stable under the
    /// arena's block bookkeeping (I-107 / 09b pitfall 16).
    ///
    /// @invariant `off` addresses `len` valid bytes on the emitting call's
    ///            `outStrArena` until the caller's enclosing `ScratchScope`
    ///            rewinds it.
    struct CopyRef {
        ArenaOffset off;
        int32_t len;
    };
}
