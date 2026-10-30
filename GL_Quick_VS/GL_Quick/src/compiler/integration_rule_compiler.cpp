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
#include "integration_rule_compiler.hpp"
#include "structural_compiler.hpp"
#include "implication_builder.hpp"
#include "definition_expansion.hpp"
#include "../compiler.hpp"
#include "../proof/worker_context.hpp"
#include "../logical_block/integration_instruction_builder.hpp"
#include "../logical_block/flat_integration_walk.hpp"
#include "../logical_block/or_introduction_rules.hpp"

#include <map>
#include <set>

namespace gl {
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param structuralCompiler Borrowed structuralCompiler component (compacts are compiled through it).
/// @param integrationBuilder Borrowed integration instruction builder (builds the definition table).
/// @return Constructed compilation component.
IntegrationRuleCompiler::IntegrationRuleCompiler(CompiledDefinitionRegistry& definitions,
    StructuralCompiler& structuralCompiler, IntegrationInstructionBuilder& integrationBuilder)
    : definitions(definitions), structuralCompiler(structuralCompiler), integrationBuilder(integrationBuilder) {}

/// @brief Compile the integration set of a registry entity at its creation
///        and store it on the entity.
///
/// @details
/// Called by `StructuralCompiler::compileExpression` right after an `and`
/// or `existence` entity is minted or reused, as the existence and OR hooks
/// do for the disintegration set. An entity that already carries its set (a
/// registry reuse, or an entity loaded from a GL binary) returns unchanged —
/// a defined no-op, never a backfill: every and/existence entity gets its set
/// at creation, and the loader asserts it present. Single-threaded seam only
/// (I-137); heap use is sanctioned here, never on the burst path.
///
/// @param name Registry name of an `and`- or `existence`-category entity
///        (asserted).
/// @return Void.
/// @invariant Afterwards the entity carries a non-empty integration set
///            whose first entry is its `OneLevel` rule.
/// @see compileIntegrationFamily — the body; LogicalEntity::integrationImplications.
void IntegrationRuleCompiler::compileIntegrationImplications(const std::string& name) {
    const auto it = definitions.compiledExpressions.find(name);
    assert(it != definitions.compiledExpressions.end()
        && (it->second.category == "and" || it->second.category == "existence")
        && "compileIntegrationImplications: name must resolve to an and or existence entity");
    if (!it->second.integrationImplications.empty()) return;
    // Copy: the compiles below insert implication entries into the same map.
    const LogicalEntity entity = it->second;
    std::vector<IntegrationImplication> family = compileIntegrationFamily(entity, true);
    assert(!family.empty() && family.front().kind == IntegrationRuleKind::OneLevel
        && "compileIntegrationImplications: the set must start with the one-level rule");
    definitions.compiledExpressions.find(name)->second.integrationImplications = std::move(family);
}

/// @brief The integration set of an `and` or `existence` entity, in the
///        order `prepareIntegration` installs it.
///
/// @details
/// The entity's definition table is built with the run-time instruction
/// builder over the entity's own tokens (`prepareIntegrationCore`, block-free
/// overload: level 0, fresh `pi_lev_0_<n>` names for the witnesses of the
/// definition tree). Then, in order:
/// 1. `OneLevel`: the entity's elements imply the entity (Case B), the
///    existence witness bound;
/// 2. for an existence, `Occupied`: the same rule with the witness replaced
///    by the extra token `u_<arity+1>`;
/// 3. for each OR premise of the one-level rule, its `OrIntroduction` rules
///    (`forEachOrIntroductionRule`, citing entry 0) followed by the
///    `DisjunctFlat` rules of each existence disjunct
///    (`forEachFlatIntegrationVariant` with the disjunct as root);
/// 4. every `Flat` variant (`forEachFlatIntegrationVariant` with the entity
///    as root), each followed by the introductions (citing that variant) and
///    disjunct rules of its OR premises not seen before.
/// Every rule is compiled through `compactOf`. The witnesses of the
/// definition tree are bound inside each compact. The premise lists are
/// copied out of the scratch table before any compile, so no scratch span is
/// held across a registry mint.
///
/// @param entity         The entity; category `and` or `existence` (asserted).
/// @param registryEntity True for an entity of the registry (the definition
///                       table resolves it by name); false for an entity
///                       outside it (the extensionality entity of a typed `=`,
///                       I-267), which the table instantiates directly.
/// @return The integration set.
/// @see compileIntegrationImplications, StructuralCompiler::ensureTypedEqualityEntity.
std::vector<IntegrationImplication> IntegrationRuleCompiler::compileIntegrationFamily(
    const LogicalEntity& entity, bool registryEntity) {
    assert((entity.category == "and" || entity.category == "existence")
        && "compileIntegrationFamily: entity must be an and or an existence");
    const unsigned genSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : genScratchArenas().slotCount() - 1;
    const unsigned strSlot = (gl::WorkerContext::coreId() >= 0)
        ? static_cast<unsigned>(gl::WorkerContext::coreId())
        : scratchArenas().slotCount() - 1;
    ScratchArena& genArena = genScratchArenas().forSlot(genSlot);
    ScratchArena& strArena = scratchArenas().forSlot(strSlot);
    ScratchScope genScope(genArena);
    ScratchScope strScope(strArena);

    WorkInstruction table(&genArena);
    const std::string goal = gl::DefinitionExpansion::removeUPrefixFromArguments(entity.signature);
    int nameCounter = 0;
    integrationBuilder.prepareIntegrationCore(StrSpan(entity.signature), table, 0, nameCounter,
        StrSpan(goal), false, registryEntity ? nullptr : &entity);
    assert(table.entityCount() > 0 && equalSpans(table.signature(0), StrSpan(entity.signature))
        && "compileIntegrationFamily: the definition table must start with the entity");
    const int32_t tokenCount = static_cast<int32_t>(ce::getArgs(entity.signature).size());

    const auto spansToStrings = [](const StrSpan* spans, int32_t n) {
        std::vector<std::string> out;
        out.reserve(static_cast<std::size_t>(n));
        for (int32_t i = 0; i < n; ++i) out.emplace_back(spans[i].ptr, static_cast<std::size_t>(spans[i].len));
        return out;
    };
    const auto elementsOf = [&](int32_t e) {
        std::vector<std::string> out;
        for (int32_t j = 0; j < table.elemCount(e); ++j) {
            const StrSpan elem = table.elemAt(e, j);
            out.emplace_back(elem.ptr, static_cast<std::size_t>(elem.len));
        }
        return out;
    };
    const auto entityWithSignature = [&](const std::string& signature) {
        for (int32_t e = 0; e < table.entityCount(); ++e)
            if (equalSpans(table.signature(e), StrSpan(signature))) return e;
        return int32_t(-1);
    };

    std::vector<IntegrationImplication> family;
    const std::vector<std::string> oneLevel = elementsOf(0);
    family.push_back({ compactOf(oneLevel, entity.signature, tokenCount), IntegrationRuleKind::OneLevel, -1 });

    if (entity.category == "existence") {
        // The witness is the one argument of the elements that is not an
        // owner token; a witness fixed by a marker replay becomes the extra
        // token u_<arity+1>.
        std::string witness;
        for (const std::string& element : oneLevel) {
            for (const std::string& argument : ce::getArgs(element)) {
                if (argument.rfind("u_", 0) == 0) continue;
                assert((witness.empty() || witness == argument)
                    && "compileIntegrationFamily: an existence with more than one witness");
                witness = argument;
            }
        }
        assert(!witness.empty() && "compileIntegrationFamily: an existence without a witness");
        const std::map<std::string, std::string> occupy = {
            { witness, "u_" + std::to_string(tokenCount + 1) } };
        std::vector<std::string> occupied;
        for (const std::string& element : oneLevel) occupied.push_back(ce::replaceKeysInString(element, occupy));
        family.push_back({ compactOf(occupied, entity.signature, tokenCount + 1),
            IntegrationRuleKind::Occupied, -1 });
    }

    // The root family: introductions and disjunct rules of every OR premise,
    // each OR premise once, then the flat variants.
    std::set<std::string> orPremisesSeen;
    const auto prepareOrPremises = [&](const std::vector<std::string>& premises, int32_t cited) {
        for (const std::string& orPremise : premises) {
            if (!orPremisesSeen.insert(orPremise).second) continue;
            std::vector<std::vector<std::string>> introductions;
            const int32_t disjunctCount = forEachOrIntroductionRule(definitions, StrSpan(orPremise), strArena,
                [&](StrSpan, const StrSpan* introPremises, int32_t introPremiseN, StrSpan) {
                    introductions.push_back(spansToStrings(introPremises, introPremiseN));
                });
            if (disjunctCount == 0) continue;
            for (const std::vector<std::string>& introPremises : introductions) {
                family.push_back({ compactOf(introPremises, orPremise, tokenCount),
                    IntegrationRuleKind::OrIntroduction, cited });
            }
            const int32_t orEntity = entityWithSignature(orPremise);
            assert(orEntity >= 0 && equalSpans(table.category(orEntity), StrSpan("or", 2))
                && "compileIntegrationFamily: a compiled OR premise has no OR entity in the table");
            for (const std::string& disjunct : elementsOf(orEntity)) {
                const int32_t disjunctEntity = entityWithSignature(disjunct);
                if (disjunctEntity < 0 || !equalSpans(table.category(disjunctEntity), StrSpan("existence", 9)))
                    continue;
                std::vector<std::vector<std::string>> disjunctVariants;
                forEachFlatIntegrationVariant(table, disjunctEntity, definitions.configuredNames(),
                    [&](const StrSpan* variantPremises, int32_t variantPremiseN) {
                        disjunctVariants.push_back(spansToStrings(variantPremises, variantPremiseN));
                    });
                for (const std::vector<std::string>& variant : disjunctVariants) {
                    family.push_back({ compactOf(variant, disjunct, tokenCount),
                        IntegrationRuleKind::DisjunctFlat, -1 });
                }
            }
        }
    };
    prepareOrPremises(oneLevel, 0);

    std::vector<std::vector<std::string>> variants;
    forEachFlatIntegrationVariant(table, 0, definitions.configuredNames(),
        [&](const StrSpan* variantPremises, int32_t variantPremiseN) {
            variants.push_back(spansToStrings(variantPremises, variantPremiseN));
        });
    for (const std::vector<std::string>& variant : variants) {
        const int32_t index = static_cast<int32_t>(family.size());
        family.push_back({ compactOf(variant, entity.signature, tokenCount), IntegrationRuleKind::Flat, -1 });
        prepareOrPremises(variant, index);
    }

    table.strings.release();
    table.data.release();
    table.elemCol.release();
    return family;
}

/// @brief Expand a compact implication into its rule through its registry
///        entry, in the compact's own argument spelling.
///
/// @details
/// The implication entity's signature tokens are replaced by the compact's
/// arguments positionally (token-boundary, simultaneous), and the rule is the
/// full-bind reconstruction of its elements (the last element is the head).
/// Bound variables keep the registry's names.
///
/// @param compact A compact instance `(implication<N>[args])`; its core must
///        resolve to an implication entity with at least two elements
///        (asserted).
/// @return The rule.
/// @see compactOf — the round-trip check that uses it.
std::string IntegrationRuleCompiler::expandIntegrationCompact(const std::string& compact) const {
    const LogicalEntity* le = definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(compact)));
    assert(le != nullptr && le->category == "implication" && le->elements.size() >= 2
        && "expandIntegrationCompact: a compact must resolve to an implication entity");
    const std::vector<std::string> formal = ce::getArgs(le->signature);
    const std::vector<std::string> actual = ce::getArgs(compact);
    assert(formal.size() == actual.size()
        && "expandIntegrationCompact: compact arity differs from its implication entity");
    std::map<std::string, std::string> substitution;
    for (std::size_t i = 0; i < formal.size(); ++i) substitution[formal[i]] = actual[i];
    std::vector<std::string> premises;
    for (std::size_t i = 0; i + 1 < le->elements.size(); ++i)
        premises.push_back(ce::replaceKeysInString(le->elements[i], substitution));
    const std::string head = ce::replaceKeysInString(le->elements.back(), substitution);
    return gl::ImplicationBuilder::reconstructImplicationFullBind(premises, head);
}

/// @brief Whether two implications are equal up to the names of their bound
///        variables.
///
/// @details
/// Each bound variable is renamed, in order of first appearance in a binder
/// list `>[...]`, to the canonical name `#bound<k>`; the two canonical texts
/// are then compared byte for byte. The free arguments, the premise order and
/// the binder placement must therefore agree exactly.
///
/// @param a The first implication.
/// @param b The second implication.
/// @return True iff the canonical forms are identical.
bool IntegrationRuleCompiler::sameUpToBoundNames(const std::string& a, const std::string& b) {
    const auto canonical = [](const std::string& text) {
        std::map<std::string, std::string> renaming;
        std::size_t at = 0;
        while ((at = text.find(">[", at)) != std::string::npos) {
            const std::size_t close = text.find(']', at + 2);
            assert(close != std::string::npos && "sameUpToBoundNames: unterminated binder list");
            std::size_t start = at + 2;
            while (start < close) {
                std::size_t comma = text.find(',', start);
                if (comma == std::string::npos || comma > close) comma = close;
                const std::string name = text.substr(start, comma - start);
                if (!name.empty() && renaming.find(name) == renaming.end())
                    renaming[name] = "#bound" + std::to_string(renaming.size() + 1);
                start = comma + 1;
            }
            at = close + 1;
        }
        return ce::replaceKeysInString(text, renaming);
    };
    return canonical(a) == canonical(b);
}

/// @brief Compile one integration rule into a compact over the owner's
///        tokens.
///
/// @details
/// The rule is the full-bind reconstruction of @p premises and @p head; it is
/// compiled through `StructuralCompiler::compileImplicationToCompact`
/// (dedup-or-mint) and restored to the owner's `u_` token names. Every
/// argument of the compact must be an owner token `u_1..u_<tokenCount>`, and
/// the compact must expand back to exactly the rule up to bound names — a
/// registry reuse that hands back an implication with permuted premises
/// asserts here (Rule 19).
///
/// @param premises   The rule premises over the owner's tokens.
/// @param head       The rule head.
/// @param tokenCount The number of owner tokens the compact may name.
/// @return The compact `(implication<N>[u_…])`.
std::string IntegrationRuleCompiler::compactOf(const std::vector<std::string>& premises,
    const std::string& head, int32_t tokenCount) {
    assert(!premises.empty() && "compactOf: an integration rule without premises");
    const std::string rule = gl::ImplicationBuilder::reconstructImplicationFullBind(premises, head);
    const std::string compact = gl::DefinitionExpansion::prefixArgumentsWithU(
        structuralCompiler.compileImplicationToCompact(rule));
    for (const std::string& argument : ce::getArgs(compact)) {
        assert(argument.size() > 2 && argument.rfind("u_", 0) == 0
            && "compactOf: a compact argument is not an owner token");
        const int token = std::stoi(argument.substr(2));
        assert(token >= 1 && token <= tokenCount && "compactOf: a compact argument outside the owner tokens");
        (void)token;
    }
    assert(sameUpToBoundNames(expandIntegrationCompact(compact), rule)
        && "compactOf: the compiled compact does not reproduce its integration rule");
    return compact;
}

} // namespace gl
