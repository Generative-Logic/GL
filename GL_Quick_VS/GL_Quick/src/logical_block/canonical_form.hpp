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
/// @brief Result of `canonicalFormAtScope` — the class-canonical
        ///        form of a statement plus the equality1 justifier tail for
        ///        the substitutions actually applied.
        ///
        /// @details Stack-and-parameter value (the I-116 span discipline):
        /// `text` and each `eqJust` span alias bytes on the caller's out
        /// arena, valid until the caller's enclosing scratch-scope rewind.
        struct CanonicalForm {
            StrSpan text{};       ///< Canonical form; aliases the input when `!changed`.
            bool changed = false; ///< Whether any substitution applied.
            int32_t eqN = 0;      ///< Justifier count below.
            /// `(=[member,canonical])` fresh builds on the caller's out
            /// arena, one per applied substitution pair, ascending by
            /// member name.
            StrSpan eqJust[ExecutionParameters::MAX_ARITY];
            /// Per justifier: whether the applied equality is a LOCAL
            /// statement of this LB (`appliedEqualityIsLocal`) — the
            /// rewritten form is a local derivation iff any entry is true.
            bool eqLocal[ExecutionParameters::MAX_ARITY] = {};
        };
}
