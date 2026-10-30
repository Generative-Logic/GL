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
#include "existence_rule_compiler.hpp"
#include "../compilation_state.hpp"
#include "structural_compiler.hpp"
#include "implication_builder.hpp"
#include "definition_expansion.hpp"
#include "expression_parser.hpp"
#include "../compiler.hpp"
#include <numeric>
#include <fstream>

namespace gl {
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param structuralCompiler Borrowed structuralCompiler component.
/// @return Constructed compilation component.
ExistenceRuleCompiler::ExistenceRuleCompiler(CompiledDefinitionRegistry& definitions, StructuralCompiler& structuralCompiler)
    : definitions(definitions), structuralCompiler(structuralCompiler) {}
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
void ExistenceRuleCompiler::compileExistenceImplications(const std::string& exName) {
    const auto it = definitions.compiledExpressions.find(exName);
    assert(it != definitions.compiledExpressions.end() && it->second.category == "existence"
        && "compileExistenceImplications: name must resolve to an existence entity");
    // Already carrying its list (registry reuse / reload re-registration):
    // a defined state, not a failure.
    if (!it->second.implications.empty()) return;
    // Copies: the compiles below insert implication entries into the same
    // map (node-stable, but the copy is the honest form).
    const std::vector<std::string> elements = it->second.elements;
    const std::size_t arity = ce::getArgs(it->second.signature).size();
    // A non-[left, right] shape (test-only) carries no compacts — a defined
    // state, the formula's zero.
    if (existenceImplicationCount(it->second) == 0) {
        return;
    }

    std::vector<std::string> compacts;
    compacts.reserve(2);
    for (std::size_t i = 0; i < 2; ++i) {
        // [0]: left -> !right; [1]: right -> !left. negate, not a blind "!"
        // prefix: a negated element's negation is its bare positive core
        // (I-175).
        const std::vector<std::string> premises = { elements[i] };
        const std::string head = gl::DefinitionExpansion::negate(elements[1 - i]);
        const std::string rule = gl::ImplicationBuilder::reconstructImplicationFullBind(premises, head);
        // The registry stores the bound variable as `1` — the very token
        // compileImplicationToCompact's u_ strip produces from `u_1` — so
        // it is renamed to the compiler's placeholder first (the same step
        // the `!(>` compile branch takes).
        const std::tuple<std::string, int, std::string> renamed =
            ExpressionParser::renameLastRemoved(rule, definitions.variableCounter);
        definitions.variableCounter++;
        const std::string compact =
            gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(std::get<0>(renamed)));
        // Premise and head together carry every existence token (the bound
        // variable is the only non-u_ argument), so the compact spans the
        // existence's full arity.
        assert(ce::getArgs(compact).size() == arity
            && "compileExistenceImplications: compact does not span the existence's tokens");
        compacts.push_back(compact);
    }
    // Re-find: the compiles above insert implication entries into the same
    // map (node-stable, but the re-fetch is the honest form).
    definitions.compiledExpressions.find(exName)->second.implications = std::move(compacts);
}

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
bool ExistenceRuleCompiler::isTypingDirectionCompact(StrSpan compact) const {
    const LogicalEntity* le = definitions.compiledEntity(extractExpressionUniversalSpan(compact));
    assert(le != nullptr && le->category == "implication" && le->elements.size() == 2
        && "isTypingDirectionCompact: an existence-implication compact resolves to a two-element implication");
    const StrSpan premise(le->elements[0]);
    if (!equalSpans(extractExpressionUniversalSpan(premise), StrSpan("in", 2))) return false;
    StrSpan args[ExecutionParameters::MAX_ARITY];
    const int32_t argN = getArgsSpans(premise, args, ExecutionParameters::MAX_ARITY);
    // The bound variable is the one premise argument without the u_ prefix;
    // `in[bound, S]` is the typing predicate, `in[u_x, bound]` is not.
    return argN == 2 && !(args[0].len >= 2 && args[0].ptr[0] == 'u' && args[0].ptr[1] == '_');
}

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
int32_t ExistenceRuleCompiler::existenceImplicationCount(const LogicalEntity& le) const {
    if (le.elements.size() != 2) return 0;
    for (const std::string& element : le.elements) {
        const std::string core =
            extractExpressionUniversalSpan(StrSpan(element)).toStdString();
        const LogicalEntity* coreLe = definitions.compiledEntity(StrSpan(core));
        if (coreLe != nullptr && isNumberedCompilationName(core, coreLe->category))
            return 0;
    }
    return 2;
}

} // namespace gl
