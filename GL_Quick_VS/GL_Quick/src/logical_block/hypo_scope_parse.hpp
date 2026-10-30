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

        /// @brief Parse result of one hypo validity name — the parent-scope
        ///        slice plus the deduplicated `_var<i>_` payload variables.
        ///
        /// @details
        /// Filled by @ref parseHypoScopeVars. Every span ALIASES the parsed
        /// `validityName`'s bytes — the caller guarantees those bytes stay
        /// mint-stable for as long as the spans are read (`reactToHypo`
        /// parses an owned per-row `ScratchString` copy for exactly this
        /// reason). Stack-and-parameter object only, never stored (I-116).
        ///
        /// `vars` holds the variables in INSERTION (payload) order after a
        /// content dedup; consumers read it as a membership set (`contains` /
        /// `empty` only), so the set order of the retired
        /// `std::set<std::string>` is unobservable and no sort is kept. The
        /// capacity is `ExecutionParameters::MAX_ARITY`: the only producer of
        /// hypo validity names (`disintegrateExprHypothetically`) writes a
        /// deduplicated subset of ONE expression's `ce::getArgs` arguments,
        /// and arity is capped at `MAX_ARITY` (`getArgsSpans` asserts the same
        /// bound).
        ///
        /// @invariant Spans point into the parsed name; they die with it.
        /// @see parseHypoScopeVars, reactToHypo,
        ///      `prover.cpp::disintegrateExprHypothetically` — the producer.
        struct HypoScopeParse {
            /// @brief The scope prefix before the `_boundary_` that opens the
            ///        hypo payload — `[0, posBoundary)` of the parsed name.
            StrSpan parentValidity;
            /// @brief Deduplicated payload variables, insertion order.
            StrSpan vars[ExecutionParameters::MAX_ARITY];
            /// @brief Number of live entries in `vars`.
            int32_t varCount = 0;
        };
}
