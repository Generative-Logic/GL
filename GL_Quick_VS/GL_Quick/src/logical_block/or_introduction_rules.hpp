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
#include "../compiler/implication_builder.hpp"

namespace gl {

/// @brief Visit the introduction implications `(>[open](D_k)(OR))` of one
///        positive OR premise, one per introducible flattened disjunct.
///
/// @details
/// "One true disjunct states the disjunction" (D-237). The OR premise
/// resolves through the compiled-definition map; a premise that is negated,
/// atomic or of another category is not an OR premise, and nothing is
/// visited — a defined classification, never a failure. The disjuncts come
/// from the compiled-map twin of `flattenOrLeaves`: each OR level substitutes
/// its signature arguments by the instance arguments and recurses into
/// OR-category elements only, so a nested OR contributes its own leaves. A
/// positive conjunction disjunct unfolds into its ordered conjuncts (I-264);
/// signed leaves stay intact. Every argument keeps its `u_` status: leaf and
/// head are fed to `reconstructImplicationFullBindScratch` in raw element
/// form, so `u_` arguments stay free and only the open arguments bind.
///
/// A disjunct is introducible when its ordered premises contain every open
/// (non-`u_`) argument of @p orPremise. Otherwise "disjunct implies OR"
/// would leave a head argument unbound (I-4) and no closed introduction
/// exists, so the disjunct is not visited — a defined classification, never
/// a failure (D-559). An OR inside a
/// definition carries its other arguments as `u_` parameters, so all its
/// disjuncts are introducible; the OR premise of a closed theorem may have
/// disjuncts that are not (H4a's `or5[k,x,P,n,a]`: `in2[k,x,P]` lacks n and
/// a, the equality conjunction lacks P).
///
/// The sink runs synchronously once per introducible disjunct, in disjunct order, with the
/// disjunct, its ordered premises and the reconstructed rule; every span
/// lives on @p arena (or in @p orPremise) and stays valid until the caller
/// rewinds that arena. Nothing is minted.
///
/// Callers: the implication branch of
/// `DisintegrationProcessor::disintegrateExprCoreView` (the rules of an
/// installed compact implication's OR premise, D-237), the compile-time
/// integration set (`IntegrationRuleCompiler::compileIntegrationImplications`,
/// the rules of a root integration goal's OR premise, D-492) and
/// `IntegrationProcessor::installRootIntegrationFamily` (an installed
/// introduction's disjunct, for its double-marker demand).
///
/// @tparam Sink Callable `void(StrSpan disjunct, const StrSpan* premises,
///              int32_t premiseCount, StrSpan rule)`.
/// @param definitions Compiled-definition registry (read only).
/// @param orPremise   One premise element in raw form, possibly `u_`-prefixed.
/// @param arena       Caller's string-tier arena receiving every built span;
///                    no scope is opened here.
/// @param sink        Receives each introducible disjunct's introduction rule.
/// @return Number of flattened disjuncts, introducible or not; 0 when
///         @p orPremise is not a positive compiled OR.
/// @invariant Every rule reaching the sink binds each open head argument
///            through its premises (I-4); a compiled OR has at least two
///            leaves (asserted); nested OR and conjunction definitions are
///            acyclic (asserted).
/// @see DisintegrationProcessor::flattenOrLeaves,
///      ImplicationBuilder::reconstructImplicationFullBindScratch.
template <class Sink>
inline int32_t forEachOrIntroductionRule(const CompiledDefinitionRegistry& definitions,
    StrSpan orPremise, ScratchArena& arena, Sink&& sink) {
    if (orPremise.len < 1 || orPremise.ptr[0] != '(') return 0;
    const LogicalEntity* orLe = definitions.compiledEntity(extractExpressionUniversalSpan(orPremise));
    if (orLe == nullptr || orLe->category != "or") return 0;

    // Flatten the OR premise's contiguous nested-OR leaves with the premise's
    // argument instantiation. Each level substitutes its signature arguments
    // by the instance arguments (fresh bytes on the string tier), recursing
    // into OR-category elements only.
    ScratchString leafHold[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    StrSpan leaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    int32_t leafN = 0;
    const auto flattenCompiledOr = [&](const LogicalEntity& node, StrSpan instance,
                                       const auto& self) -> void {
        StrSpan sigArgs[ExecutionParameters::MAX_ARITY];
        const int32_t sigN = getArgsSpans(StrSpan(node.signature), sigArgs, ExecutionParameters::MAX_ARITY);
        StrSpan instArgs[ExecutionParameters::MAX_ARITY];
        const int32_t instN = getArgsSpans(instance, instArgs, ExecutionParameters::MAX_ARITY);
        assert(sigN == instN && "or-intro: instance arity differs from compiled or");
        StrReplacement pairs[ExecutionParameters::MAX_ARITY];
        for (int32_t a = 0; a < sigN; ++a) {
            pairs[a].key = sigArgs[a];
            pairs[a].value = instArgs[a];
        }
        for (const std::string& rawElem : node.elements) {
            const ScratchString subst = replaceKeysScratch(arena, StrSpan(rawElem), pairs, sigN);
            const StrSpan substSpan(subst);
            const LogicalEntity* childLe = (substSpan.len >= 1 && substSpan.ptr[0] == '(')
                ? definitions.compiledEntity(extractExpressionUniversalSpan(substSpan))
                : nullptr;
            if (childLe != nullptr && childLe->category == "or") {
                self(*childLe, substSpan, self);
            } else {
                assert(leafN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS
                    && "or-intro: leaf count exceeds cap");
                leafHold[leafN] = subst;
                leaves[leafN] = StrSpan(leafHold[leafN]);
                ++leafN;
            }
        }
    };
    flattenCompiledOr(*orLe, orPremise, flattenCompiledOr);
    assert(leafN >= 2 && "or-intro: compiled or has fewer than two leaves");

    for (int32_t li = 0; li < leafN; ++li) {
        StrSpan premises[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        int32_t premiseN = 0;
        StrSpan conjunctionAncestors[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const auto flattenConjunction = [&](StrSpan expression, int depth, auto&& self) -> void {
            assert(depth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
            const StrSpan core = extractExpressionUniversalSpan(expression);
            const LogicalEntity* definition = expression.ptr[0] == '!'
                ? nullptr : definitions.compiledEntity(core);
            if (definition != nullptr && definition->category == "and") {
                for (int i = 0; i < depth; ++i)
                    assert(!equalSpans(conjunctionAncestors[i], core)
                        && "or-intro: cyclic conjunction definition");
                conjunctionAncestors[depth] = core;
                StrSpan formal[ExecutionParameters::MAX_ARITY], actual[ExecutionParameters::MAX_ARITY];
                const int count = getArgsSpans(StrSpan(definition->signature), formal,
                    ExecutionParameters::MAX_ARITY);
                assert(count == getArgsSpans(expression, actual, ExecutionParameters::MAX_ARITY));
                StrReplacement substitution[ExecutionParameters::MAX_ARITY];
                for (int i = 0; i < count; ++i) substitution[i] = {formal[i], actual[i]};
                assert(!definition->elements.empty());
                for (const auto& element : definition->elements) {
                    const ScratchString child = replaceKeysScratch(arena,
                        StrSpan(element), substitution, count);
                    self(StrSpan(child), depth + 1, self);
                }
            } else {
                assert(premiseN < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
                premises[premiseN++] = expression;
            }
        };
        flattenConjunction(leaves[li], 0, flattenConjunction);
        // Only a disjunct whose premise chain contains every changeable head
        // argument has a closed introduction (I-4); any other disjunct is not
        // introducible (D-559).
        bool introducible = true;
        {
            StrSpan headArgs[ExecutionParameters::MAX_ARITY];
            const int32_t headArgN = getArgsSpans(orPremise, headArgs, ExecutionParameters::MAX_ARITY);
            for (int32_t ha = 0; ha < headArgN && introducible; ++ha) {
                if (startsWithSpan(headArgs[ha], "u_", 2)) continue;
                bool inLeaf = false;
                for (int32_t pi = 0; pi < premiseN && !inLeaf; ++pi) {
                    StrSpan leafArgs[ExecutionParameters::MAX_ARITY];
                    const int32_t leafArgN = getArgsSpans(premises[pi], leafArgs, ExecutionParameters::MAX_ARITY);
                    for (int32_t la = 0; la < leafArgN && !inLeaf; ++la)
                        inLeaf = equalSpans(leafArgs[la], headArgs[ha]);
                }
                introducible = inLeaf;
            }
        }
        if (!introducible) continue;
        const ScratchString rule =
            ImplicationBuilder::reconstructImplicationFullBindScratch(arena, premises, premiseN, orPremise);
        sink(leaves[li], static_cast<const StrSpan*>(premises), premiseN, StrSpan(rule));
    }
    return leafN;
}

} // namespace gl
