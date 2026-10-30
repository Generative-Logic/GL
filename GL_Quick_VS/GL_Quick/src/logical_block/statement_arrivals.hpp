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

/// @brief Stages membership changes for existing OR and subproof consumers.
/// @details Storage, allocation and invocation order remain with callers.
class StatementArrivals {
public:

/// @brief Queue current-state reconciliation for a relevant known-fact key.
/// @details An unbuilt index will read current known state during bootstrap.
/// A built index only watches signed cores that still have active OR sources.
/// @param mb Owning block and its source/index state.
/// @param key Original expression and validity whose known membership changed.
/// @return Void.
static void stageOrFactChange(Memory& mb, int64_t key);

/// @brief Stage a known-row change that is an event of a demand-built OR
///        head (D-597).
///
/// @details
/// A deferred OR is built when a fact refutes one of its disjuncts. A row
/// with the shape of a registered OR head's refutation template
/// (`DemandBuiltRuleRegistry::hasOrTrigger`), or a row an event waits for
/// (`Memory::deferredOrWaiting`, keyed by the missing premise text), is
/// staged for `RejectedSourceReplay::drainDeferredOrEvents`; any other row
/// is no event and stays unstaged. Called beside the OR exclusion staging
/// for every known-row publication and erasure; the drain reads the row's
/// current membership.
///
/// @param mb  Owning block.
/// @param key Packed `(original, scope)` of the changed row.
/// @return Nothing; appends to `Memory::pendingDeferredOrEvents` only.
/// @see RejectedSourceReplay::drainDeferredOrEvents.
static void stageDeferredOrEvent(Memory& mb, int64_t key);

/// @brief Stage one newly known row for the implication-gate drain
        ///        (D-406).
        ///
        /// @details
        /// Called beside every write that makes a row known
        /// (`stageWatchArrival`, `publishKnownStatementKey`). The drain at
        /// step 4b of `standardProcessing` matches the staged rows against
        /// the parked gate keys. While no implication is parked the arrival
        /// wakes nothing and is not staged: a later park checks knownness
        /// itself, so an arrival that precedes every park is already seen.
        ///
        /// @param mb  Owning block.
        /// @param key Packed `(original, scope)` of the newly known row.
        /// @invariant Appends to `pendingSubproofArrivals` only.
        /// @see drainSubproofArrivals, parkImplicationSubproof.
/// @return Nothing; updates the specified storage in place.
static void stageSubproofArrival(Memory& mb, int64_t key);

/// @brief Stage one newly known row for the contradiction-scope demand
        ///        drain (D-560).
        ///
        /// @details
        /// Called beside every write that makes a row known, next to
        /// `stageSubproofArrival`. The drain at step 4b of
        /// `standardProcessing` files the row in the demand's fact index and
        /// matches it against the rule heads that negate it, for every
        /// registered contradiction scope that sees the row. While no
        /// contradiction scope is registered the arrival is not staged: a
        /// scope registered later matches every fact it sees when the drain
        /// first serves it.
        ///
        /// @param mb  Owning block.
        /// @param key Packed `(original, scope)` of the newly known row.
        /// @invariant Appends to `pendingContradictionArrivals` only.
        /// @see ContradictionDemand::drainContradictionDemands,
        ///      stageSubproofArrival.
/// @return Nothing; updates the specified storage in place.
static void stageContradictionArrival(Memory& mb, int64_t key);

};

/// @brief Stage one newly known row for the implication-gate drain
        ///        (D-406).
        ///
        /// @details
        /// Called beside every write that makes a row known
        /// (`stageWatchArrival`, `publishKnownStatementKey`). The drain at
        /// step 4b of `standardProcessing` matches the staged rows against
        /// the parked gate keys. While no implication is parked the arrival
        /// wakes nothing and is not staged: a later park checks knownness
        /// itself, so an arrival that precedes every park is already seen.
        ///
        /// @param mb  Owning block.
        /// @param key Packed `(original, scope)` of the newly known row.
        /// @invariant Appends to `pendingSubproofArrivals` only.
        /// @see drainSubproofArrivals, parkImplicationSubproof.
/// @return Nothing; updates the specified storage in place.
inline void StatementArrivals::stageSubproofArrival(Memory& mb, int64_t key) {
            if (mb.overallHashMemory.rejectedMapSubproof.empty()) return;
            mb.pendingSubproofArrivals.push_back(key);
        }

/// @brief Stage one newly known row for the contradiction-scope demand
        ///        drain (D-560).
        ///
        /// @details
        /// Called beside every write that makes a row known, next to
        /// `stageSubproofArrival`. The drain at step 4b of
        /// `standardProcessing` files the row in the demand's fact index and
        /// matches it against the rule heads that negate it, for every
        /// registered contradiction scope that sees the row. While no
        /// contradiction scope is registered the arrival is not staged: a
        /// scope registered later matches every fact it sees when the drain
        /// first serves it.
        ///
        /// @param mb  Owning block.
        /// @param key Packed `(original, scope)` of the newly known row.
        /// @invariant Appends to `pendingContradictionArrivals` only.
        /// @see ContradictionDemand::drainContradictionDemands,
        ///      stageSubproofArrival.
/// @return Nothing; updates the specified storage in place.
inline void StatementArrivals::stageContradictionArrival(Memory& mb, int64_t key) {
            if (mb.lbMemory.contradictionScopes.empty()) return;
            mb.pendingContradictionArrivals.push_back(key);
        }
}
