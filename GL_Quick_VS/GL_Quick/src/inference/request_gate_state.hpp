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







        /// @brief Running summary of the request gates over a premise prefix.
        ///
        /// @details
        /// The three map-independent gates are all order-free conjunctions over
        /// the premises, so each one reduces to a summary a premise can be folded
        /// into one at a time: the hypothesis scope, the distinct scopes of the
        /// premises that a main-scope anchor exemption does not cover, and the
        /// distinct secondary variables. That makes the gates resumable in the
        /// same way the normalized key is, which is what the request generator's
        /// grow search needs — the prefix is folded once per stack pop and only
        /// the appended premise is folded per position.
        ///
        /// `nonExemptScopes` saturates at 2: the hypothesis gate only ever asks
        /// whether the non-exempt premises share ONE scope, so a third distinct
        /// scope carries no extra information.
        ///
        /// `seenSecondary` is sized by `(MAX_EXPRESSIONS + 2) * MAX_ARITY` — the
        /// pairing merge's widest request — and the distinct count is 2-3 in
        /// practice, so the linear insert-if-absent beats a hash set's per-node
        /// allocation on the hottest burst path. Overflow is a loud assert, never
        /// a truncation.
        ///
        /// @invariant `nonExemptScopes == 0` implies every premise so far was a
        ///            main-scope anchor; `== 1` implies they all share
        ///            `nonExemptValidityId`.
        /// @see `foldExprIntoRequestGates` — the per-premise writer.
        /// @see `requestGateStateAccepts` — the verdict over a finished summary.
        struct RequestGateState {
            static constexpr int kMaxSeenSecondary =
                (ExecutionParameters::MAX_EXPRESSIONS + 2)
                    * ExecutionParameters::MAX_ARITY;

            int32_t hypoFound;
            NameId  hypoValidityId;
            int32_t nonExemptScopes;
            NameId  nonExemptValidityId;
            NameId  seenSecondary[kMaxSeenSecondary];
            int32_t secondaryCount;
        };
}
