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
#include "compiled_definition_registry.hpp"

namespace gl {
class CompiledDefinitionRegistry;
class StructuralCompiler;

/// @brief Compile the implication family of named existence definitions.
/// @details Composes through definition interfaces without retaining a prover.
class ExistenceRuleCompiler {
public:
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param structuralCompiler Borrowed structuralCompiler component.
/// @return Constructed compilation component.
ExistenceRuleCompiler(CompiledDefinitionRegistry& definitions, StructuralCompiler& structuralCompiler);
/// @brief Compile an existence's "existence implications" — the two
        /// rules a negated existence `!(existence<N>[args])` disintegrates
        /// into, `left → !right` and `right → !left` — into
        /// `LogicalEntity::implications` as `(implication<N>[u_…])` compacts
        /// over the existence's own `u_` tokens
        /// (D-310).
        ///
        /// @details
        /// An existence entity encodes `!(>[b](left)!(right))`; its negation
        /// is the universal `(>[b](left)!(right))`, i.e. the rule
        /// `left → !right` and, contrapositively, `right → !left`. For each
        /// of the two the chain is `reconstructImplicationFullBind` over the
        /// registry elements (the head negated with double-negation
        /// cancellation, I-175); the bound variable is renamed through
        /// `renameLastRemoved` to the compiler's placeholder (the registry
        /// stores it as `1`, the very token `compileImplicationToCompact`'s
        /// `u_` strip produces from `u_1`); the rule is compiled through the
        /// mail-broadcast compaction door `compileImplicationToCompact`
        /// (dedup-or-mint) and restored to the existence's token names with
        /// `prefixArgumentsWithU`. List order: `[0]` = `left → !right`,
        /// `[1]` = `right → !left`.
        ///
        /// Idempotent: an existence that already carries a list returns
        /// unchanged (a registry reuse or a reload re-registering the same
        /// existence). A registry existence with other than two elements
        /// (test-only shapes) carries none
        /// (`existenceImplicationCount` = 0). Barrier seam only —
        /// heap use is sanctioned here, never on the burst path.
        ///
        /// @param exName Registry name of an existence-category entity
        ///        (asserted).
        /// @return (void)
        /// @invariant Single-threaded seam only (I-137). Afterwards the list
        ///            holds `existenceImplicationCount(entity)`
        ///            compacts, each with the existence's full arity
        ///            (premise and head together carry every token,
        ///            asserted).
        /// @see `compileExpression` — the `!(>` mint-site caller;
        ///      `disintegrateExprCore2` — the consumer (negated-existence
        ///      block).
void compileExistenceImplications(const std::string& exName);
/// @brief Whether an existence-implication compact is the TYPING
        /// direction of its existence — the rule `in[k,S] → !body(k)` whose
        /// premise is the bound variable's bare typing predicate.
        ///
        /// @details
        /// The negated-existence door of `disintegrateExprCore2` instantiates
        /// every compact of the existence's `implications` list except this
        /// one (D-382): the typing direction is a
        /// blind universal instantiation of the negated body over every typed
        /// term (one negated row per term, consumed by no rule); the body
        /// direction `body(k) → !in[k,S]` is demand-shaped and is the rule
        /// every proof on record uses. The compiled list, the binaries and
        /// `existenceImplicationCount` are untouched — only the instantiation
        /// is gated. The premise element is `elements[0]` of the compact's
        /// implication entity; the bound variable is its one argument without
        /// the `u_` prefix (`1` in the registry), so `in[1,u_1]` is the typing
        /// predicate and `in[u_1,1]` is not. Registry read only, no
        /// allocation (burst path, I-138).
        ///
        /// @param compact The compact instance or template
        ///        `(implication<N>[…])`; must resolve to a two-element
        ///        implication entity (asserted).
        /// @return True iff the premise is the bound variable's typing
        ///         predicate.
        /// @see `compileExistenceImplications` — the list's producer;
        ///      `disintegrateExprCore2` — the door.
bool isTypingDirectionCompact(StrSpan compact) const;
/// @brief The length of a COMPLETE existence-implication list for an
        /// existence entity: 2 for the compiler's `[left, right]` shape when
        /// BOTH elements are named operators, 0 otherwise.
        ///
        /// @details
        /// The named-only rule (I-209): an existence whose element core is a
        /// numbered compilation name — a generated `and<N>`, a nested
        /// `existence<N>`, an `implication<N>`, an `or<N>`
        /// (`isNumberedCompilationName`) — gets no existence implications, so
        /// its negation is a stored statement only. With the compacts, a
        /// negated nested existence instantiated its body over every typed
        /// term at every quantifier level (the D1 cube). The Peano
        /// predecessor existence (`in`, `in2`), `preorder` (`in`, `in3`) and
        /// every other existence over named predicates keep their two rules.
        /// The disintegrator's completeness tripwire and the compile gate
        /// share this one formula. An element core
        /// without a registry entry is a named atom (config predicate).
        ///
        /// @param le The existence entity.
        /// @return 2 or 0.
        /// @see `compileExistenceImplications`; `isNumberedCompilationName`.
int32_t existenceImplicationCount(const LogicalEntity& le) const;

private:
CompiledDefinitionRegistry& definitions;
StructuralCompiler& structuralCompiler;
};
} // namespace gl
