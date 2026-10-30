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

        /// @brief Result of one `openImplicationSubproof` call.
        ///
        /// @details
        /// `Known`: the implication's clean compact is already a known
        /// statement at the preparing scope (up to its complete classes), so
        /// no subproof is needed. `Prepared`: the scope for the same payload
        /// was prepared before (goal-qualified on main) and is not opened a
        /// second time. `Refused`: the implication names a construction the
        /// recursion block built over a listed witness
        /// (`Memory::builtOnListedWitnesses`,
        /// I-293); nothing is prepared, parked or
        /// opened. `Parked`: an and of the instruction that lists the
        /// implication has a gate element not yet known at the scope or an
        /// ancestor; the implication waits in `rejectedMapSubproof` under
        /// that element (D-406). `Opened`: the
        /// boundary scope was pushed and seeded with the premises and the
        /// head.
        enum class ImplicationSubproofOutcome : uint8_t {
            Known,
            Prepared,
            Refused,
            Parked,
            Opened
        };
}
