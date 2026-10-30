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
#include "../memory_infra/str_ops.hpp"

namespace gl {

/// @brief Classifies numbered compact expression names without compiling or processing them.
/// @details Storage, allocation and invocation order remain with callers.
class CompactExpressionSyntax {
public:

/// @brief Whether a span has the compact-implication shape
        ///        `(implication<digits>[…]` — the one carrier shape the
        ///        canonical gate applies to.
        ///
        /// @details Reads the numbered-name prefix only; does not validate or process the full expression.
        /// @param s The expression span.
        /// @return `true` for `(implication` followed by a digit.
        /// @see `installExpandedImplication`, `addExprToMemoryBlock`.
static bool isCompactImplicationSpan(StrSpan s);

/// @brief Whether a span has the or-compact statement shape
        ///        `(or<digits>[…` — the one shape the deposit door's
        ///        or-uniqueness gate applies to.
        ///
        /// @details
        /// Matches exactly the positive registry-or spelling the cohort
        /// machinery opens on. The De-Morgan door's `!(op[args])` deposits
        /// and or-shaped GOALS (status 2, which leave the door before the
        /// gate) never match, so both keep their current paths.
        ///
        /// @details Reads the numbered-name prefix only; does not validate or process the full expression.
        /// @param s The expression span.
        /// @return `true` for `(or` followed by a digit.
        /// @see `addExprToMemoryBlock` — the gate's one seam;
        ///      `orEquiRepresentativeRecorded`.
static bool isOrCompactSpan(StrSpan s);
};

/// @brief Whether a span has the compact-implication shape
        ///        `(implication<digits>[…]` — the one carrier shape the
        ///        canonical gate applies to.
        ///
        /// @details Reads the numbered-name prefix only; does not validate or process the full expression.
        /// @param s The expression span.
        /// @return `true` for `(implication` followed by a digit.
        /// @see `installExpandedImplication`, `addExprToMemoryBlock`.
inline bool CompactExpressionSyntax::isCompactImplicationSpan(StrSpan s) {
            return s.len > 12 && startsWithSpan(s, "(implication", 12)
                && s.ptr[12] >= '0' && s.ptr[12] <= '9';
        }

/// @brief Whether a span has the or-compact statement shape
        ///        `(or<digits>[…` — the one shape the deposit door's
        ///        or-uniqueness gate applies to.
        ///
        /// @details
        /// Matches exactly the positive registry-or spelling the cohort
        /// machinery opens on. The De-Morgan door's `!(op[args])` deposits
        /// and or-shaped GOALS (status 2, which leave the door before the
        /// gate) never match, so both keep their current paths.
        ///
        /// @details Reads the numbered-name prefix only; does not validate or process the full expression.
        /// @param s The expression span.
        /// @return `true` for `(or` followed by a digit.
        /// @see `addExprToMemoryBlock` — the gate's one seam;
        ///      `orEquiRepresentativeRecorded`.
inline bool CompactExpressionSyntax::isOrCompactSpan(StrSpan s) {
            return s.len > 3 && startsWithSpan(s, "(or", 3)
                && s.ptr[3] >= '0' && s.ptr[3] <= '9';
        }
}
