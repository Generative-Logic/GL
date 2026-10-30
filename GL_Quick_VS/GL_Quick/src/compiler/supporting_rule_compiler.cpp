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
#include "supporting_rule_compiler.hpp"
#include "structural_compiler.hpp"
#include "implication_builder.hpp"
#include "definition_expansion.hpp"
#include "../compiler.hpp"
#include "../proof/worker_context.hpp"
#include "../logical_block/integration_instruction_builder.hpp"
#include "../logical_block/disintegration_shots.hpp"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <string_view>

namespace gl {
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param structuralCompiler Borrowed structuralCompiler component.
/// @return Constructed compilation component.
SupportingRuleCompiler::SupportingRuleCompiler(CompiledDefinitionRegistry& definitions, StructuralCompiler& structuralCompiler)
    : definitions(definitions), structuralCompiler(structuralCompiler) {}
/// @brief Prepare every binder-covering contraposition in compiled families.
        /// @details Uses only single-premise disintegration products. Conjunction
        /// families retain their source order and every covering subset; generated
        /// auxiliaries never become new defining elements or generation families.
        /// @return Void.
        /// @invariant Compilation and canonical reference completion precede workers.
void SupportingRuleCompiler::compileContrapositionFamilies() {
    struct Source {
        std::string compact;
        std::string premise;
        std::string head;
        std::set<std::string> bound;
    };
    // The entries present now, by entity address — forEachEntryWithAdded
    // tells the entries this pass adds from them — and the owners to prepare.
    std::vector<const LogicalEntity*> present;
    present.reserve(definitions.compiledExpressions.size());
    std::set<std::string> groupedCarriers;
    std::vector<std::string> owners;
    for (const auto& [name, entity] : definitions.compiledExpressions) {
        present.push_back(&entity);
        if (!entity.contrapositionPrepared && !entity.contrapositionAuxiliary
            && (entity.category == "and" || entity.category == "implication"))
            owners.push_back(name);
    }
    const auto variables = [](const std::string& expression) {
        std::set<std::string> result;
        for (const auto& argument : ce::getArgs(expression))
            if (argument.rfind("u_", 0) != 0) result.insert(argument);
        return result;
    };
    std::set<std::string> coveredCompacts;
    std::vector<std::string> incompleteCompacts;
    for (const auto& name : owners) {
        const LogicalEntity owner = definitions.compiledExpressions.at(name);
        std::vector<Source> sources;
        const auto collect = [&](const std::string& instance, const auto& self) -> void {
            if (instance.empty() || instance.front() != '(') return;
            const auto core = extractExpressionUniversalSpan(StrSpan(instance)).toStdString();
            const auto& entity = definitions.compiledExpressions.at(core);
            const auto formal = ce::getArgs(entity.signature);
            const auto actual = ce::getArgs(instance);
            assert(formal.size() == actual.size());
            std::map<std::string, std::string> substitution;
            for (std::size_t i = 0; i < formal.size(); ++i) substitution[formal[i]] = actual[i];
            if (entity.category == "and") {
                for (const auto& element : entity.elements)
                    self(ce::replaceKeysInString(element, substitution), self);
            } else if (entity.category == "implication" && entity.elements.size() == 2) {
                if (gl::SupportingRuleCompiler::isContrapositionTypingExclusion(entity)) return;
                if (gl::SupportingRuleCompiler::isContrapositionUncoveredDefinition(entity, definitions.expressionsFromConfig.count(core) != 0))
                    return;
                Source source{instance,
                    ce::replaceKeysInString(entity.elements[0], substitution),
                    ce::replaceKeysInString(entity.elements[1], substitution), {}};
                // The existing product-of-disintegration classification requires
                // a fixed parameter in the premise; unparameterized theorems are
                // outside this mechanism.
                const auto premiseArguments = ce::getArgs(source.premise);
                bool product = false;
                std::map<std::string, std::string> renaming;
                for (const auto& argument : premiseArguments) {
                    if (argument.rfind("u_", 0) == 0) product = true;
                    else if (!renaming.count(argument))
                        renaming[argument] = "contra_variable_" + std::to_string(renaming.size() + 1);
                }
                if (!product) return;
                const auto premiseVariables = variables(source.premise);
                const auto headVariables = variables(source.head);
                assert(std::includes(premiseVariables.begin(), premiseVariables.end(),
                    headVariables.begin(), headVariables.end())
                    && "contraposition: original premise omits a bound variable");
                source.premise = ce::replaceKeysInString(source.premise, renaming);
                source.head = ce::replaceKeysInString(source.head, renaming);
                source.bound = variables(source.premise);
                sources.push_back(std::move(source));
            }
        };
        collect(owner.signature, collect);
        std::vector<ContrapositionTemplate> prepared;
        std::set<std::pair<std::string, std::string>> seen;
        std::vector<bool> covered(sources.size(), false);
        for (std::size_t first = 0; first < sources.size(); ++first) {
            bool earlier = false;
            for (std::size_t j = 0; j < first; ++j)
                if (sources[j].premise == sources[first].premise) earlier = true;
            if (earlier) continue;
            std::vector<std::size_t> family;
            for (std::size_t j = first; j < sources.size(); ++j)
                if (sources[j].premise == sources[first].premise) family.push_back(j);
            std::vector<std::size_t> selected;
            const auto enumerate = [&](std::size_t position, const auto& self) -> void {
                if (position < family.size()) {
                    selected.push_back(family[position]);
                    self(position + 1, self);
                    selected.pop_back();
                    self(position + 1, self);
                    return;
                }
                if (selected.empty()) return;
                std::set<std::string> unionVariables;
                for (auto index : selected) {
                    const auto vars = variables(sources[index].head);
                    unionVariables.insert(vars.begin(), vars.end());
                }
                if (unionVariables != sources[first].bound) return;
                std::string head = sources[selected.back()].head;
                for (std::size_t i = selected.size() - 1; i > 0; --i)
                    head = "(&" + sources[selected[i - 1]].head + head + ")";
                std::string compiledHead = structuralCompiler.compileExpression(
                    head);
                // Conjunction compilation returns bare instance arguments.
                // Restore the fixed parameters before reconstructing binders;
                // the distinct bound-variable names cannot collide with them.
                std::map<std::string, std::string> fixedArguments;
                for (auto index : selected)
                    for (const auto& argument : ce::getArgs(sources[index].head))
                        if (argument.rfind("u_", 0) == 0)
                            fixedArguments[argument.substr(2)] = argument;
                compiledHead = ce::replaceKeysInString(compiledHead, fixedArguments);
                assert(variables(compiledHead) == unionVariables
                    && "contraposition: head compilation changed the bound-variable set");
                const std::string sourceText = gl::ImplicationBuilder::reconstructImplicationFullBind(
                    {sources[first].premise}, compiledHead);
                const std::string reverseText = gl::ImplicationBuilder::reconstructImplicationFullBind(
                    {gl::DefinitionExpansion::negate(compiledHead)}, gl::DefinitionExpansion::negate(sources[first].premise));
                ContrapositionTemplate record;
                for (auto index : selected) {
                    record.sources.push_back(sources[index].compact);
                    covered[index] = true;
                }
                record.sourceRule = selected.size() == 1 ? record.sources.front()
                    : gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(sourceText));
                record.reverseRule = gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(reverseText));
                if (selected.size() > 1)
                    groupedCarriers.insert(extractExpressionUniversalSpan(StrSpan(record.sourceRule)).toStdString());
                if (seen.emplace(record.sourceRule, record.reverseRule).second)
                    prepared.push_back(std::move(record));
            };
            enumerate(0, enumerate);
        }
        for (std::size_t i = 0; i < sources.size(); ++i) {
            const auto core = extractExpressionUniversalSpan(StrSpan(sources[i].compact)).toStdString();
            if (covered[i]) coveredCompacts.insert(core);
            else if (owner.category == "and") {
                std::cerr << "Uncovered contraposition family: " << name << " source "
                    << sources[i].compact << " premise " << sources[i].premise << '\n';
                assert(false && "contraposition: disintegration family cannot cover every binder");
            } else incompleteCompacts.push_back(core);
        }
        auto& target = definitions.compiledExpressions.at(name);
        target.contrapositives = std::move(prepared);
        target.contrapositionPrepared = true;
    }
    // A grouped source is a real compact carrier. Its own prepared reversal
    // reproduces the rule pair after mail, canonical rewriting or reload.
    for (const auto& name : groupedCarriers) {
        const LogicalEntity group = definitions.compiledExpressions.at(name);
        if (!group.contrapositives.empty()) continue;
        assert(group.category == "implication" && group.elements.size() == 2);
        std::map<std::string, std::string> renaming;
        for (const auto& argument : ce::getArgs(group.elements[0]))
            if (argument.rfind("u_", 0) != 0 && !renaming.count(argument))
                renaming[argument] = "contra_variable_" + std::to_string(renaming.size() + 1);
        const std::string premise = ce::replaceKeysInString(group.elements[0], renaming);
        const std::string head = ce::replaceKeysInString(group.elements[1], renaming);
        const std::string reverse = gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(
            gl::ImplicationBuilder::reconstructImplicationFullBind({gl::DefinitionExpansion::negate(head)}, gl::DefinitionExpansion::negate(premise))));
        definitions.compiledExpressions.at(name).contrapositives.push_back(
            ContrapositionTemplate{{group.signature}, group.signature, reverse});
        definitions.compiledExpressions.at(name).contrapositionPrepared = true;
    }
    // An incomplete standalone source must be covered by a family; a split
    // source may have been prepared in an earlier completion seam, so the
    // sources of every prepared record count. The covered set is read only by
    // this check, so it is built only when there is a source to check.
    if (!incompleteCompacts.empty()) {
        for (const auto& [name, entity] : definitions.compiledExpressions) {
            (void)name;
            for (const auto& record : entity.contrapositives)
                for (const auto& source : record.sources)
                    coveredCompacts.insert(extractExpressionUniversalSpan(StrSpan(source)).toStdString());
        }
        for (const auto& name : incompleteCompacts) {
            if (!coveredCompacts.count(name))
                std::cerr << "Uncovered standalone contraposition source: " << name << '\n';
            assert(coveredCompacts.count(name)
                && "contraposition: unary source has no binder-covering disintegration family");
        }
    }
    forEachEntryWithAdded(definitions.compiledExpressions, present,
        [](const std::string&, LogicalEntity& entity, bool added) {
            if (!added) return;
            entity.contrapositionAuxiliary = true;
            entity.contrapositionPrepared = true;
        });
    compileConjunctionRefutations();
    compileSupportingImplications();
}

/// @brief Compile fixed-parameter refutations of conjunction members.
        /// @details Prepare direct and nested member templates independently of
        /// runtime facts. Mark eligible prepared reversals for scoped discovery.
        /// Defining elements and integration instructions remain unchanged.
        /// @return Void.
        /// @invariant Called only at single-threaded compilation seams.
        /// @see compileContrapositionFamilies.
void SupportingRuleCompiler::compileConjunctionRefutations() {
    std::vector<std::string> owners;
    std::vector<const LogicalEntity*> present;
    present.reserve(definitions.compiledExpressions.size());
    for (const auto& [name, entity] : definitions.compiledExpressions) {
        present.push_back(&entity);
        if (entity.category == "and" && !entity.conjunctionRefutationsPrepared)
            owners.push_back(name);
    }
    for (const auto& name : owners) {
        const LogicalEntity owner = definitions.compiledExpressions.at(name);
        std::vector<std::pair<std::string, std::string>> prepared;
        std::set<std::string> seen;
        const auto collect = [&](const std::string& conjunction, const auto& self) -> void {
            const auto& entity = definitions.compiledExpressions.at(
                extractExpressionUniversalSpan(StrSpan(conjunction)).toStdString());
            assert(entity.category == "and");
            const auto formal = ce::getArgs(entity.signature);
            const auto actual = ce::getArgs(conjunction);
            assert(formal.size() == actual.size());
            std::map<std::string, std::string> substitution;
            for (std::size_t i = 0; i < formal.size(); ++i) substitution[formal[i]] = actual[i];
            const auto elements = entity.elements;
            for (const auto& element : elements) {
                const std::string member = ce::replaceKeysInString(element, substitution);
                if (!seen.insert(member).second) continue;
                const std::string rule = gl::ImplicationBuilder::reconstructImplicationFullBind(
                    {gl::DefinitionExpansion::negate(member)}, gl::DefinitionExpansion::negate(owner.signature));
                const std::string compact = gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(rule));
                prepared.emplace_back(member, compact);
                const auto* child = definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(member)));
                if (member.front() == '(' && child != nullptr && child->category == "and")
                    self(member, self);
            }
        };
        collect(owner.signature, collect);
        auto& destination = definitions.compiledExpressions.at(name);
        destination.conjunctionRefutations = std::move(prepared);
        destination.conjunctionRefutationsPrepared = true;
    }
    auto& compiled = definitions.compiledExpressions;
    forEachEntryWithAdded(compiled, present,
        [&compiled](const std::string&, LogicalEntity& entity, bool added) {
            if (added) {
                entity.contrapositionAuxiliary = true;
                entity.contrapositionPrepared = true;
            }
            // The helper is found by a view of its name (the map's transparent
            // comparator), no name copy.
            for (const auto& record : entity.conjunctionRefutations) {
                const StrSpan helperName = extractExpressionUniversalSpan(StrSpan(record.second));
                const auto helper = compiled.find(std::string_view(
                    helperName.ptr, static_cast<std::size_t>(helperName.len)));
                assert(helper != compiled.end()
                    && "compileConjunctionRefutations: a refutation names no compiled helper");
                helper->second.supportingImplicationAuxiliary = true;
            }
        });
}

/// @brief Compile guarded helpers for negated immediate conjunction premises.
        /// @details Preserve fixed parameters and complete premise binding; generated
        /// helpers do not recursively seed either auxiliary preparation family.
        /// Unary owners install direct compositions with proof-only intermediates.
        /// @return Void.
        /// @invariant Called only at single-threaded compilation seams.
void SupportingRuleCompiler::compileSupportingImplications() {
    std::vector<std::string> owners;
    std::vector<const LogicalEntity*> present;
    present.reserve(definitions.compiledExpressions.size());
    for (const auto& [name, entity] : definitions.compiledExpressions) {
        present.push_back(&entity);
        if (entity.category == "implication" && !entity.supportingImplicationsPrepared
            && !entity.supportingImplicationAuxiliary) owners.push_back(name);
    }
    for (const auto& name : owners) {
        // Compiler reuse can make a previously registered compact a helper.
        if (definitions.compiledExpressions.at(name).supportingImplicationAuxiliary) continue;
        const LogicalEntity owner = definitions.compiledExpressions.at(name);
        assert(owner.elements.size() >= 2);
        const bool compose = owner.elements.size() == 2;
        std::vector<std::string> prepared, composed;
        std::set<std::string> seen;
        for (std::size_t p = 0; p + 1 < owner.elements.size(); ++p) {
            const std::string& premise = owner.elements[p];
            if (premise.empty() || premise.front() != '!') continue;
            const auto* definition = definitions.compiledEntity(extractExpressionUniversalSpan(StrSpan(premise)));
            if (definition == nullptr || definition->category != "and") continue;
            const LogicalEntity conjunction = *definition;
            const auto formal = ce::getArgs(conjunction.signature);
            const auto actual = ce::getArgs(premise);
            assert(formal.size() == actual.size() && !conjunction.elements.empty());
            std::map<std::string, std::string> mapping;
            for (std::size_t i = 0; i < formal.size(); ++i) mapping[formal[i]] = actual[i];
            std::vector<std::string> parts;
            std::set<std::string> covered;
            for (const auto& element : conjunction.elements) {
                parts.push_back(ce::replaceKeysInString(element, mapping));
                for (const auto& argument : ce::getArgs(parts.back())) covered.insert(argument);
            }
            for (const auto& argument : actual)
                assert(argument.rfind("u_", 0) == 0 || covered.count(argument) != 0);
            // Keep bound names distinct from the bare forms of fixed u_N ports
            // while the compact compiler normalizes its numbered arguments.
            std::map<std::string, std::string> boundNames;
            for (const auto& argument : actual)
                if (argument.rfind("u_", 0) != 0 && !boundNames.count(argument))
                    boundNames[argument] = "support_variable_" + std::to_string(boundNames.size() + 1);
            for (auto& part : parts) part = ce::replaceKeysInString(part, boundNames);
            const std::string helperHead = ce::replaceKeysInString(premise, boundNames);
            for (std::size_t selected = 0; selected < parts.size(); ++selected) {
                auto chain = parts;
                chain[selected] = gl::DefinitionExpansion::negate(chain[selected]);
                const std::string compact = gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(
                    gl::ImplicationBuilder::reconstructImplicationFullBind(chain, helperHead)));
                if (seen.insert(compact).second) {
                    prepared.push_back(compact);
                    if (compose) {
                        for (const auto& argument : ce::getArgs(owner.elements.back()))
                            assert(argument.rfind("u_", 0) == 0 || covered.count(argument) != 0);
                        const std::string finalHead = ce::replaceKeysInString(owner.elements.back(), boundNames);
                        const std::string direct = gl::DefinitionExpansion::prefixArgumentsWithU(structuralCompiler.compileImplicationToCompact(
                            gl::ImplicationBuilder::reconstructImplicationFullBind(chain, finalHead)));
                        composed.push_back(direct);
                        auto& directRule = definitions.compiledExpressions.at(
                            extractExpressionUniversalSpan(StrSpan(direct)).toStdString());
                        directRule.supportingImplicationAuxiliary = true;
                        directRule.contrapositionAuxiliary = true;
                        directRule.contrapositionPrepared = true;
                    }
                }
                auto& helper = definitions.compiledExpressions.at(extractExpressionUniversalSpan(StrSpan(compact)).toStdString());
                helper.supportingImplicationAuxiliary = true;
                helper.contrapositionAuxiliary = true;
                helper.contrapositionPrepared = true;
            }
        }
        auto& destination = definitions.compiledExpressions.at(name);
        assert(composed.empty() || composed.size() == prepared.size());
        destination.supportingImplications = std::move(prepared);
        destination.composedSupportingImplications = std::move(composed);
        destination.supportingImplicationsPrepared = true;
    }
    forEachEntryWithAdded(definitions.compiledExpressions, present,
        [](const std::string&, LogicalEntity& entity, bool added) {
            if (!added) return;
            entity.supportingImplicationAuxiliary = true;
            entity.supportingImplicationsPrepared = true;
            entity.contrapositionAuxiliary = true;
            entity.contrapositionPrepared = true;
        });
}

/// @brief Recognize a public implication definition whose head omits a premise binder.
        /// @details A configured definition whose whole body is one implication
        /// (`primeseq`: `P(i)=x` implies `prime(x)`) installs its forward rule.
        /// When the head does not mention every variable the premise binds (the
        /// index `i`), the reversed rule would leave that variable unbound, so no
        /// binder-covering contrapositive exists. Contraposition preparation skips
        /// such a definition (D-468); its
        /// forward rule stays installed. Generated implication compacts are not
        /// public definitions and keep the I-260 coverage assert.
        /// @param entity Compiled candidate implication.
        /// @param publicDefinition Whether the entity's name is a configured
        /// definition (`expressionsFromConfig`, filled before generated compacts
        /// extend `coreExpressionMap`).
        /// @return True for precisely a public single-premise implication definition
        /// whose head leaves a premise-bound variable uncovered.
bool SupportingRuleCompiler::isContrapositionUncoveredDefinition(const LogicalEntity& entity, bool publicDefinition) {
    if (!publicDefinition || entity.category != "implication" || entity.elements.size() != 2)
        return false;
    std::set<std::string> headBound;
    for (const auto& argument : ce::getArgs(entity.elements[1]))
        if (argument.rfind("u_", 0) != 0) headBound.insert(argument);
    for (const auto& argument : ce::getArgs(entity.elements[0]))
        if (argument.rfind("u_", 0) != 0 && headBound.count(argument) == 0) return true;
    return false;
}

/// @brief Prepare the disintegration package of every existence entity
///        (I-333).
///
/// @details
/// Walks a name-ordered snapshot of the existence entities without a
/// prepared package and stores on each the package
/// `computeDisintegrationPackage` derives. Idempotent: a prepared entity is
/// skipped. Single-threaded seam only (I-137): it runs beside
/// `compileContrapositionFamilies` at the compile end, at the prover's
/// initialization and at the reduced-OR seam before every burst, so every
/// existence carries its package before any disintegration reads it.
///
/// @return Void.
/// @invariant Afterwards every existence entity has `shotsPrepared`.
/// @see computeDisintegrationPackage,
///      DisintegrationProcessor::disintegrateExprCoreView — the consumer.
void SupportingRuleCompiler::compileDisintegrationPackages() {
    std::vector<std::string> names;
    for (const auto& [name, entity] : definitions.compiledExpressions)
        if (entity.category == "existence" && !entity.shotsPrepared) names.push_back(name);
    for (const std::string& name : names) {
        std::vector<std::string> binders;
        std::vector<std::string> placeholders;
        std::vector<DisintegrationShot> shots;
        computeDisintegrationPackage(definitions.compiledExpressions.at(name), binders, placeholders, shots);
        LogicalEntity& entity = definitions.compiledExpressions.at(name);
        entity.shotBinders = std::move(binders);
        entity.shotPlaceholders = std::move(placeholders);
        entity.shots = std::move(shots);
        entity.shotsPrepared = true;
    }
}

/// @brief The disintegration package of one existence entity: its internal
///        universals fired once on its products
///        (I-333).
///
/// @details
/// The entity's definition tree is unfolded with the run-time instruction
/// builder (`prepareIntegrationCore`, block-free, level 0: its bound
/// variables become `pi_lev_0_<n>`) and walked as the disintegrator walks
/// it: an existence opens a binder for its one new bound variable, an
/// existence or a conjunction contributes its elements as products and
/// descends into the elements that are themselves existences or
/// conjunctions, and an implication element is a universal whose rule is
/// rebuilt from its compact entity. Every bound variable is spelled
/// `u_pi_lev_0_<n>` in every text, as the disintegrator spells an
/// enclosing witness, so a universal binds only its own variables. Each
/// universal fires once on all products through `shootUniversal`. A shot
/// belongs to this entity when its universal and premises lie on one binder
/// path and do not all lie inside one nested existence (that existence's own
/// package holds those); it attaches to the innermost binder of the path.
/// Duplicate shots are dropped; order is the walk order.
///
/// @param entity       An existence entity of the registry (asserted).
/// @param binders      Receives the binder templates; binder 0 is the
///                     entity's own signature.
/// @param placeholders Receives the placeholder of each binder.
/// @param shots        Receives the shots.
/// @return Void.
/// @invariant Mints into no registry; a pure function of the registry.
/// @see shootUniversal, compileDisintegrationPackages.
void SupportingRuleCompiler::computeDisintegrationPackage(const LogicalEntity& entity,
    std::vector<std::string>& binders, std::vector<std::string>& placeholders,
    std::vector<DisintegrationShot>& shots) const {
    assert(entity.category == "existence" && "computeDisintegrationPackage: an existence entity");
    binders.clear();
    placeholders.clear();
    shots.clear();
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
    IntegrationInstructionBuilder builder(definitions);
    const std::string goal = gl::DefinitionExpansion::removeUPrefixFromArguments(entity.signature);
    int nameCounter = 0;
    builder.prepareIntegrationCore(StrSpan(entity.signature), table, 0, nameCounter, StrSpan(goal), true);
    assert(table.entityCount() > 0 && equalSpans(table.signature(0), StrSpan(entity.signature))
        && "computeDisintegrationPackage: the definition table must start with the entity");

    const auto text = [](StrSpan span) { return std::string(span.ptr, static_cast<std::size_t>(span.len)); };
    const auto isPlaceholder = [](StrSpan token) { return startsWithSpan(token, "pi_lev_", 7); };
    // The disintegrator's spelling of an enclosing witness: u_<name>.
    const auto spellBound = [&](const std::string& source) {
        std::map<std::string, std::string> renaming;
        collectExprTokens(StrSpan(source), [&](StrSpan token) {
            if (isPlaceholder(token)) renaming.emplace(text(token), "u_" + text(token));
        });
        return renaming.empty() ? source : ce::replaceKeysInString(source, renaming);
    };
    const auto entityOf = [&](const std::string& element) -> int32_t {
        for (int32_t e = 0; e < table.entityCount(); ++e)
            if (equalSpans(table.signature(e), StrSpan(element))) return e;
        return -1;
    };

    struct Item {
        std::string text;
        std::vector<int32_t> chain;
    };
    std::vector<Item> products;
    std::vector<Item> universals;
    std::vector<std::string> universalCompacts;
    std::set<std::string> bound;
    const auto walk = [&](int32_t e, std::vector<int32_t> chain, const auto& self) -> void {
        const StrSpan category = table.category(e);
        if (equalSpans(category, StrSpan("existence", 9))) {
            std::string fresh;
            for (int32_t j = 0; j < table.elemCount(e); ++j) {
                StrSpan args[ExecutionParameters::MAX_ARITY];
                const int32_t n = getArgsSpans(table.elemAt(e, j), args, ExecutionParameters::MAX_ARITY);
                for (int32_t a = 0; a < n; ++a) {
                    if (!isPlaceholder(args[a]) || bound.count(text(args[a])) != 0) continue;
                    assert((fresh.empty() || fresh == text(args[a]))
                        && "computeDisintegrationPackage: an existence with more than one bound variable");
                    fresh = text(args[a]);
                }
            }
            if (!fresh.empty()) {
                bound.insert(fresh);
                chain.push_back(static_cast<int32_t>(binders.size()));
                binders.push_back(spellBound(text(table.signature(e))));
                placeholders.push_back("u_" + fresh);
            }
        }
        for (int32_t j = 0; j < table.elemCount(e); ++j) {
            const std::string element = text(table.elemAt(e, j));
            const int32_t child = entityOf(element);
            const StrSpan childCategory = child >= 0 ? table.category(child) : StrSpan();
            if (equalSpans(childCategory, StrSpan("implication", 11))) {
                std::vector<std::string> premises;
                for (int32_t k = 0; k + 1 < table.elemCount(child); ++k)
                    premises.push_back(spellBound(text(table.elemAt(child, k))));
                const std::string head = spellBound(text(table.elemAt(child, table.elemCount(child) - 1)));
                universals.push_back({ gl::ImplicationBuilder::reconstructImplicationFullBind(premises, head), chain });
                universalCompacts.push_back(spellBound(element));
                continue;
            }
            products.push_back({ spellBound(element), chain });
            if (equalSpans(childCategory, StrSpan("existence", 9)) || equalSpans(childCategory, StrSpan("and", 3)))
                self(child, chain, self);
        }
    };
    walk(0, {}, walk);
    // An existence that binds no variable has no witness path at run time
    // (the disintegrator walks its elements in place): no binder, no shot.
    // Binder 0 is the entity's own whenever it binds one.
    if (binders.empty() || binders.front() != spellBound(entity.signature)) {
        binders.clear();
        placeholders.clear();
        table.strings.release();
        table.data.release();
        table.elemCol.release();
        return;
    }

    std::vector<StrSpan> productSpans;
    productSpans.reserve(products.size());
    for (const Item& product : products) productSpans.emplace_back(product.text);
    const auto isPrefix = [](const std::vector<int32_t>& prefix, const std::vector<int32_t>& chain) {
        return prefix.size() <= chain.size() && std::equal(prefix.begin(), prefix.end(), chain.begin());
    };
    std::set<std::string> seen;
    for (std::size_t u = 0; u < universals.size(); ++u) {
        ScratchScope shotScope(strArena);
        constexpr int32_t kShotCapacity = 64;
        UniversalShot found[kShotCapacity];
        const int32_t n = shootUniversal(StrSpan(universals[u].text), productSpans.data(),
            static_cast<int32_t>(productSpans.size()), strArena, found, kShotCapacity);
        for (int32_t s = 0; s < n; ++s) {
            std::vector<const std::vector<int32_t>*> chains{ &universals[u].chain };
            for (int32_t p = 0; p < found[s].premiseN; ++p)
                chains.push_back(&products[static_cast<std::size_t>(found[s].premiseProduct[p])].chain);
            const std::vector<int32_t>* longest = chains.front();
            const std::vector<int32_t>* shortest = chains.front();
            for (const std::vector<int32_t>* chain : chains) {
                if (chain->size() > longest->size()) longest = chain;
                if (chain->size() < shortest->size()) shortest = chain;
            }
            bool onePath = true;
            for (const std::vector<int32_t>* chain : chains) onePath = onePath && isPrefix(*chain, *longest);
            // Off one binder path the head's names are never bound together;
            // inside one nested existence the shot is that existence's own.
            if (!onePath || shortest->size() >= 2) continue;
            DisintegrationShot shot;
            shot.universal = universalCompacts[u];
            for (int32_t p = 0; p < found[s].premiseN; ++p)
                shot.premises.push_back(products[static_cast<std::size_t>(found[s].premiseProduct[p])].text);
            shot.head = found[s].head.toStdString();
            shot.binder = longest->back();
            std::string key = shot.universal + '\x01' + shot.head;
            for (const std::string& premise : shot.premises) key += '\x01' + premise;
            if (seen.insert(key).second) shots.push_back(std::move(shot));
        }
    }
    table.strings.release();
    table.data.release();
    table.elemCol.release();
}

} // namespace gl
