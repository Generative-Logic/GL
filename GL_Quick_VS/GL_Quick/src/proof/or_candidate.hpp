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
namespace gl {

        /// @brief OR-theorem candidate — a pair of dual implications waiting
        /// for both halves to be proved before an OR theorem can be emitted.
        ///
        /// @details
        /// An OR theorem `(a ∨ b)` is canonically proved by establishing
        /// both `!a → b` and `!b → a`. When the prover discovers either
        /// half it records an `OrCandidate` keyed on the canonical pair
        /// in `orCandidates`; when the second half lands, both
        /// `existenceProved` and `companionProved` flip to `true` and
        /// the OR theorem is emitted via `constructOrTheorem`.
        ///
        /// The companion mutex `orCandidatesMutex` protects writes during
        /// the parallel hash burst so two threads concurrently proving
        /// the two halves don't race.
        ///
        /// @see `OrCandidateProgress::checkOrCompletion` — completion flags.
        /// @see `constructOrTheorem` — final OR-theorem builder.
        /// @see [`docs/agentic_swdd/20_core_concepts/07_or_branching.md`](../../docs/agentic_swdd/20_core_concepts/07_or_branching.md)
        ///      for the full OR-branching chapter.
        struct OrCandidate {
            std::string existenceTheorem;   // !a -> b form
            std::string companionTheorem;   // !b -> a form
            bool existenceProved = false;
            bool companionProved = false;
        };
}
