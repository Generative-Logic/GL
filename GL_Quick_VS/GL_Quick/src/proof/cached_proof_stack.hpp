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
#include <map>
#include <set>
#include <string>
#include <vector>
namespace gl {



        /// @brief Cache of proof stacks for theorems whose LBs were destroyed
        /// between big iterations.
        ///
        /// @details
        /// The prover tears down certain LBs (typically integration sub-blocks
        /// or contradiction LBs) at iteration boundaries to keep memory
        /// bounded. If a theorem proven inside a torn-down LB still needs
        /// its proof stack to land in the proof graph, this cache holds
        /// the stack until export.
        ///
        /// Three slots:
        /// - `stack0` — single-chapter-proof or check-zero stack (the canonical
        ///   stack of the theorem's main proof).
        /// - `stack1` — check_induction_condition stack (induction-only).
        /// - `stack2` — induction-typing sub-proof `(in[indVar, anchor_args[0]])`.
        ///   Populated in the pre-teardown cache pass; built via
        ///   `buildStack` on the parent memoryBlock from the typing goal.
        ///   See [`docs/agentic_swdd/induction_typing_plan.md`](../../docs/agentic_swdd/induction_typing_plan.md)
        ///   and [I-18](../../docs/agentic_swdd/30_invariants.md#i-18).
        ///
        /// `cachedProofStacks` is keyed on the theorem's compiled-form text.
        struct CachedProofStack {
            std::vector<std::vector<std::string>> stack0; // single_chapter_proof or check_zero
            std::vector<std::vector<std::string>> stack1; // check_induction_condition (induction only)
            // induction typing sub-proof `(in[indVar, anchor_args[0]])` — buildStack
            // on the parent memoryBlock from the typing goal. Populated in the
            // pre-teardown cache pass; see the induction typing contract.
            std::vector<std::vector<std::string>> stack2;
        };
}
