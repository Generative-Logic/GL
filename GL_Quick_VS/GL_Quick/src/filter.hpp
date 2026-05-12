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

#include <string>

/// @file
/// @brief Counterexample (CE) filter — declarations shared with the prover
/// translation unit.
///
/// @details
/// Method bodies for the CE filter live in `filter.cpp`; they are member
/// functions of `ExpressionAnalyzer` (declared in `prover.hpp`). This header
/// is the home for CE-only types and free helpers that should not pollute
/// `prover.hpp` / `memory.hpp`.
///
/// The CE filter is the *peek-and-prune* stage of the pipeline: every
/// candidate conjecture is loaded into a synthetic LB seeded with the simple
/// facts (`files/simple_facts/simple_facts_<anchor>_<n>.txt`), and the prover
/// is run for a small budget (`numberIterationsConjectureFiltering`) to see
/// whether the conjecture's negation produces a contradiction against any of
/// the simple-fact j-copies. A confirmed contradiction marks the conjecture
/// `successful` in `contradictionTable[i]`; the filter then survives the
/// conjecture into `filtered_conjectures.txt` for the main prover stage.
///
/// @see [`docs/10_pipeline/03_ce_filter.md`](../../docs/10_pipeline/03_ce_filter.md)
///      for the full pipeline-stage chapter.

namespace gl {

    /// @brief One row of the CE-filter contradiction table.
    ///
    /// @details
    /// `ExpressionAnalyzer` holds a `std::vector<ContradictionItem>`
    /// (`contradictionTable`) keyed by conjecture index. Per-row `successful`
    /// is set `true` when the prover detects that the conjecture contradicts
    /// the simple-fact base. CE-only — referenced from `filter.cpp` and from
    /// the contradiction-handler touch-point in `prover.cpp`.
    ///
    /// @see `prover.hpp::contradictionTable`.
    struct ContradictionItem {
        std::string expr;
        bool successful;
        ContradictionItem()
            : expr(),
            successful(false){
        }
        ContradictionItem(const std::string& expr_,
            int successful_)
            : expr(expr_),
            successful(successful_) {
        }
    };

} // namespace gl
