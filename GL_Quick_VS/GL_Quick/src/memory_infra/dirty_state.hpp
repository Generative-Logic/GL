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

#include <cstdint>

namespace gl {

    /// @brief Content-change state of an LB's statified aggregate since
    ///        its last deload dump (or reload) — the skip / tail-delta /
    ///        full-rewrite decision input.
    ///
    /// @details
    /// `Clean`: RAM still equals the on-disk file set — deload releases
    /// blocks without writing. `AppendedOnly`: every mutation since the
    /// last dump was a `push_back` — the new content is the old content
    /// plus per-container tails, so deload may write a small tail file
    /// (the WAL/journal pattern). `Restructured`: an erase / clear /
    /// rebuild / potential in-place write happened — only a full
    /// canonical rewrite represents the content. States only escalate
    /// between dumps; dump and reload reset to `Clean`.
    enum class DirtyState : uint8_t {
        Clean = 0,
        AppendedOnly = 1,
        Restructured = 2,
    };

}
