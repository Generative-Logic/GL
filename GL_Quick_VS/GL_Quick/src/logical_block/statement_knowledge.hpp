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
/// @brief Query existing scoped statement knowledge.
/// @details Reads block registries without minting or recording proof decisions.
class StatementKnowledge {
public:
/// @brief Is this LB, or any of its ancestors, flagged with a `main`
    ///        contradiction (inconsistent premise set)?
    ///
    /// @details
    /// The read side of the vacuous-premise suppression: when a normal LB
    /// reaches a full contradiction at its exact `main`, its premise set is
    /// inconsistent and every theorem provable at or below it is vacuous
    /// (ex falso). `dischargeContradiction`'s normal-LB branch sets
    /// `Memory::mainContradiction` on the contradicted LB; the post-join
    /// emission gates (`drainUpdateGlobalSingleChapter`, the induction promotion in
    /// `updateGlobal`) call this walker on the producing LB and suppress the
    /// emission on a hit. The walk rides `parentMemory` to the batch root —
    /// address-stable raw pointers for the LB's whole life
    /// ([I-109](../../docs/agentic_swdd/30_invariants.md#i-109)) — so the
    /// probe is race-free at the single-threaded post-join seams where it
    /// runs. Both verdicts are defined results of the detection contract.
    ///
    /// @param lb The producing LB (self is included in the walk).
    /// @return True iff `lb` or any strict ancestor has
    ///         `mainContradiction` set.
static inline bool hasContradictedAncestor(const Memory* lb) {
        for (const Memory* p = lb; p != nullptr; p = p->parentMemory) {
            if (p->mainContradiction) return true;
        }
        return false;
    }

/// @brief The Site F ancestor scan as a shared predicate: is `origId`
    ///        known on `valId`'s scope chain?
    ///
    /// @details
    /// THE single definition of the ancestor-known dedup contract
    /// ([I-27](../../docs/agentic_swdd/30_invariants.md#i-27),
    /// D-273): an expression known at a scope is
    /// fully visible at every descendant scope through comparable-scope
    /// inheritance, so a deposit whose expression is already known at the
    /// deposit scope or an ancestor adds nothing and is refused. Callers:
    /// Site F at `addExprToMemoryBlock` entry, the firing-record mirror in
    /// `checkLocalEncodedMemoryStatic`, and the registration-door gates
    /// (`addStatement`, `applyEquivalenceClass`, the negated-equality
    /// expander's emit gate).
    ///
    /// `NameMap::ancAt(v, k)` is root-first with self at index
    /// `ancLen(v) - 1`; `includeSelf` selects whether the deposit scope
    /// itself participates (the Site F semantics) or only strict ancestors
    /// (the door gates, whose exact-scope handling is their own
    /// pre-existing gate). Every probe is non-minting; a caller holding
    /// id 0 for either name has, by I-83, a text that was never interned
    /// and therefore cannot be known — pass only minted ids.
    ///
    /// @param mb          Owning LB (registries read-only).
    /// @param origId      Minted NameMap id of the expression.
    /// @param valId       Minted NameMap id of the deposit scope.
    /// @param includeSelf True = self + strict ancestors (Site F);
    ///                    false = strict ancestors only (door gates).
    /// @return True iff a probed scope knows the expression.
    /// @invariant [I-27](../../docs/agentic_swdd/30_invariants.md#i-27) —
    ///            ancestor-scan dedup; probes never mint.
static inline bool ancestorKnown(const Memory& mb, NameId origId, NameId valId,
                              bool includeSelf) {
        assert(origId != 0 && valId != 0
            && "ancestorKnown: ids must be minted");
        const int32_t ancN = mb.nameMap.ancLen(valId);
        const int32_t limit = includeSelf ? ancN : ancN - 1;
        for (int32_t k = 0; k < limit; ++k) {
            const NameId anc = mb.nameMap.ancAt(valId, k);
            if (mb.intKnownStatements.find(StatementKey{ origId, anc })
                    != nullptr) {
                return true;
            }
        }
        return false;
    }

/// @brief The scope whose goals an OR case on @p scope's chain serves: the
    ///        deepest scope at or above @p scope holding a registered goal.
    ///
    /// @details
    /// Walks from @p scope through its ancestors to main and returns the
    /// first scope with at least one `intToBeProved` entry
    /// (D-549). Goals further up are shadowed: below an
    /// implication or integration subproof a case can help close only that
    /// subproof's goals, because the subproof exports nothing but its own
    /// result. Goals at sibling or descendant scopes are never on the chain.
    /// Zero means no scope on the chain holds a goal; an active primed
    /// contradiction block then still owns its implicit goal, the
    /// contradiction at main.
    ///
    /// @param mb    Owning LB (registries read-only).
    /// @param scope Minted scope where the walk starts (a case scope or its
    ///              cohort parent).
    /// @return The innermost goal-holding scope on the chain, or 0.
    /// @invariant Pure read; mints nothing.
    /// @see CaseProcessor::orBranchGoalsResolved,
    ///      CaseProcessor::retireRefutedGoalSubproofs.
static inline NameId innermostGoalScope(const Memory& mb, NameId scope) {
        assert(scope != 0 && "innermostGoalScope: scope must be minted");
        const int32_t goalN = mb.intToBeProved.count();
        if (goalN == 0) return 0;
        for (NameId s = scope; s != 0; s = mb.nameMap.parentOf(s)) {
            for (int32_t g = 1; g <= goalN; ++g)
                if (mb.intToBeProved.decodeKey(g).validity == s) return s;
            if (s == NameMap::MAIN_ID) break;
        }
        return 0;
    }
};
}
