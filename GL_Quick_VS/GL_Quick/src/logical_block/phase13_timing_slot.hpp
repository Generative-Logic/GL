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
#if PHASE13_DEEP_TIMING
        /// @brief Internal Phase 1 and Phase 3 timing categories.
        ///
        /// @details
        /// The active `proveKernel` phase points `phase13TimingRows` at one
        /// private row per worker. Phase helpers and `standardProcessing` add
        /// elapsed nanoseconds to the calling worker's row, avoiding atomic
        /// contention and preserving the phase's scheduling. The rows are
        /// observation-only and are folded only after the phase barrier.
        enum class Phase13TimingSlot : std::size_t {
            claimLoad,
            burstSetup,
            routingMailPull,
            externalOriginAbsorb,
            externalStatementAbsorb,
            internalOriginAbsorb,
            internalStatementAbsorb,
            internalMailClear,
            applyEquivalenceClasses,
            enrichProductsOfRecursion,
            cleanupExpressions,
            dischargeContradiction,
            dischargeToBeProved,
            dischargeContradictionScopes,
            fillMailOut,
            changedClassesClear,
            routingMailCleanup,
            reactToHypothesis,
            sanitizeToBeProved,
            drainDisprovedGoals,
            drainDeadOrBranches,
            freezeResolvedOrBranches,
            drainPendingOrReleases,
            wipeSubtrees,
            sweepAncestorKnownRows,
            quiescenceAndDumps,
            releaseClaim,
            count
        };
inline constexpr std::size_t kPhase13TimingSlotCount =
            static_cast<std::size_t>(Phase13TimingSlot::count);
#endif

}
