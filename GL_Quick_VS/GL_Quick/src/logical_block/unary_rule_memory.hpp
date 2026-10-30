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
#include "../compiler/compiled_definition_registry.hpp"
namespace gl {
/// @brief Register unary sources and qualify their existing integration demands.
/// @details Owns the registration operations while Memory retains physical storage.
class UnaryRuleMemory {
public:
/// @brief Bind definitions used by unary-rule demand matching.
/// @details Source and consumer records remain in the supplied logical block.
/// @param definitions Read-only operator configurations and type metadata.
/// @return Constructed unary source-memory interface.
UnaryRuleMemory(const CompiledDefinitionRegistry& definitions);

/// @brief Match a unary head against an existing integration demand.
        /// @details Only the witness variable becomes u_marker in the matching
        /// view; exactly one non-u_ digit variable remains. All fixed arguments,
        /// repetitions and polarity retain their ordinary classifications.
        /// @param head Rule head with its bound variables unchanged.
        /// @param demand Stored marked integration demand.
        /// @param mapping Caller array of MAX_ARITY replacements, valid on success.
        /// @param mappingCount Receives the successful replacement count.
        /// @param witness Receives the head's witness variable on success.
        /// @return True exactly when the CNFE head conditions match.
        /// @invariant Matching does not publish a fact, rule or admission key;
        /// returned views borrow the caller's head and demand storage.
bool matchUnaryAdmissionHead(StrSpan head, StrSpan demand,
            StrReplacement* mapping, int32_t& mappingCount, StrSpan& witness);

/// @brief Index a rule as a consumer for one existing demand.
        /// @details Matching rules enter the consumer-premise index once; exact
        /// demand keys retain the scope needed to test a proposed composition.
        /// @param mb Resident owning block.
        /// @param ruleId Existing composition rule ordinal.
        /// @param demandKey Existing integration admission key.
        /// @return Whether this rule can serve the demand under CNFE matching.
bool indexUnaryCompositionDemand(Memory& mb, int32_t ruleId, int64_t demandKey);

/// @brief Wake installed consumer rules for newly observed demands.
        /// @details The signed head index limits candidates. Removed demand
        /// receipts retire so a reintroduced demand creates a fresh wake event.
        /// @param mb Resident owning block with current admission maps.
        /// @return Nothing; eligible existing consumers enter the pending queue.
void wakeUnaryCompositionDemands(Memory& mb);

/// @brief Register an actual implication as a composition producer.
        /// @details Canonical endpoint identity is distinct from its exact proof citation.
        /// @param mb Owning block.
        /// @param premises Registered ordered premise run.
        /// @param premiseCount Number of registered premises.
        /// @param head Registered head.
        /// @param proof Exact installed copy expression.
        /// @param parent Exact owning rule before multiplication.
        /// @param scope Original validity scope.
        /// @param levels Sorted supporting levels.
        /// @param levelCount Number of supporting levels.
        /// @param local Whether the installation belongs to the local rule instances.
        /// @return Nothing.
void rememberUnaryImplication(Memory& mb, const StrSpan* premises, int premiseCount, StrSpan head,
            StrSpan proof, StrSpan parent, StrSpan scope, const int* levels, int32_t levelCount, bool local);

/// @brief Retire the roots owned by a removed implication.
        /// @details The next drain rebuilds derived closure from remaining roots.
        /// @param mb Owning block.
        /// @param parent Removed owning rule.
        /// @param scope Its exact validity.
        /// @return Nothing.
void forgetUnaryImplication(Memory& mb, StrSpan parent, StrSpan scope);
public:
/// @brief Enter an enrolled unary consumer into the premise join index.
        /// @details The key is the premise's signed operator signature, the
        /// value the rule id; runs are sorted and unique, so a consumer
        /// enrolled by both a demand and a goal is indexed once.
        /// @param mb Resident owning block.
        /// @param ruleId Existing unary composition rule ordinal.
        /// @return Nothing.
        /// @invariant The rule has exactly one premise.
void indexUnaryConsumerPremise(Memory& mb, int32_t ruleId);
public:
/// @brief Match a unary head against an open goal the way CNFE does.
        /// @details Same sign, operator and arity; every fixed `u_` argument
        /// equals the goal's argument at its position; exactly one bound
        /// variable, at a `(1)` definition slot.
        /// @param head Rule head with its bound variables unchanged.
        /// @param goal Goal statement as registered, without `u_` prefixes.
        /// @return Whether the head can establish the goal up to its constant.
        /// @invariant Pure predicate; creates nothing.
        /// @see checkNecessityForEquality.
bool matchUnaryGoalHead(StrSpan head, StrSpan goal);
public:
/// @brief Enroll a consumer whose head can establish an open goal.
        /// @details The goal-side twin of `indexUnaryCompositionDemand`
        /// (D-469): the rule's premise enters the join
        /// index once and the goal key stays with the rule for the open-goal
        /// check of each proposed composition.
        /// @param mb Resident owning block.
        /// @param ruleId Existing composition rule ordinal.
        /// @param goalKey Packed key of an open goal in `intToBeProved`.
        /// @return Whether CNFE matching and comparable scopes permit it.
        /// @see matchUnaryGoalHead.
bool indexUnaryCompositionGoal(Memory& mb, int32_t ruleId, int64_t goalKey);
public:
/// @brief Wake installed consumer rules for newly observed goals.
        /// @details Goals are walked in sorted order; a removed goal's receipt
        /// retires, so a goal registered again wakes its consumers again.
        /// @param mb Resident owning block with its current goal registry.
        /// @return Nothing; eligible existing consumers enter the pending queue.
void wakeUnaryCompositionGoals(Memory& mb);
public:
/// @brief Whether a goal-enrolled consumer still serves an open goal.
        /// @details One of its goals must still be registered, with a scope
        /// comparable to the composition scope whose deeper end phase 3 does
        /// not ignore.
        /// @param mb Resident owning block.
        /// @param ruleId Consumer rule ordinal.
        /// @param scopeId Scope of the proposed composition.
        /// @return Whether the composition serves at least one open goal.
bool unaryCompositionGoalOpen(Memory& mb, int32_t ruleId, NameId scopeId);
private:
const CompiledDefinitionRegistry& definitions;
};
}
