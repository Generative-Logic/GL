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

        /// @brief One deferred cross-LB admission seed (I-28 detect-and-defer
        ///        trial) — a `updateAdmissionMap` call that `updateAdmissionMap3`
        ///        resolved onto a STRICT ANCESTOR while running in a parallel
        ///        worker, captured for single-threaded replay post-join.
        ///
        /// @details
        /// A trivially-copyable POD: `ancestor` is address-stable for the LB's
        /// life (raw `Memory*`, I-109); `key` (the single `[expr]`, rule order)
        /// and `remainingArgs` (sorted-unique lex, the former `std::set` order)
        /// are `SealedSpan<SealedString>` runs on a shared `SealedPageSet`. The
        /// strings travel VERBATIM across the parallel → single-threaded join
        /// and are minted into the ancestor's interners only at the drain (the
        /// `StagedAdmissionValue` / I-83 contract — no worker-phase mint). The
        /// record itself rides the page set's record chain (`appendRecord`).
        struct DeferredAncestorAdmission {
            Memory* ancestor;
            SealedSpan<SealedString> key;
            SealedSpan<SealedString> remainingArgs;
            int maxAdmissionDepth;
            int maxSecondaryNumber;
            bool partOfRecursion;
        };
}
