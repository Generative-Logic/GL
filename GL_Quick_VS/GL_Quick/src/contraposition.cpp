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

#include "prover.hpp"

namespace gl {

/// @brief Recognize the negative-typing body-rule class excluded from reversal.
/// @details The source has one premise and denies membership of its sole bound
/// variable in a fixed set. The original rule remains installed; only auxiliary
/// contraposition generation is excluded, preserving D-382.
/// @param entity Compiled candidate source.
/// @return Whether reversing the source restores the excluded typing direction.
bool ExpressionAnalyzer::isContrapositionTypingExclusion(const LogicalEntity& entity) {
    if (entity.category != "implication" || entity.elements.size() != 2) return false;
    const auto& head = entity.elements[1];
    if (head.rfind("!(in[", 0) != 0) return false;
    const auto arguments = ce::getArgs(head);
    assert(arguments.size() == 2);
    if (arguments[0].rfind("u_", 0) == 0 || arguments[1].rfind("u_", 0) != 0) return false;
    std::set<std::string> bound;
    for (const auto& argument : ce::getArgs(entity.elements[0]))
        if (argument.rfind("u_", 0) != 0) bound.insert(argument);
    return bound.size() == 1 && bound.count(arguments[0]) == 1;
}

/// @brief Prepare all binder-covering source groups and their contrapositives.
/// @details Preparation walks definition-owned conjunction families and individual
/// unary rules. Every covering subset is retained in source order. All new
/// registry entities are auxiliary and cannot seed recursive preparation.
/// @return Void.
/// @invariant Defining elements and integration structures are not modified.
void ExpressionAnalyzer::compileContrapositionFamilies() {
    struct Source {
        std::string compact;
        std::string premise;
        std::string head;
        std::set<std::string> bound;
    };
    std::set<std::string> originalNames;
    std::set<std::string> groupedCarriers;
    std::vector<std::string> owners;
    for (const auto& [name, entity] : compiledExpressions) {
        originalNames.insert(name);
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
        const LogicalEntity owner = compiledExpressions.at(name);
        std::vector<Source> sources;
        const auto collect = [&](const std::string& instance, const auto& self) -> void {
            if (instance.empty() || instance.front() != '(') return;
            const auto core = extractExpressionUniversalSpan(StrSpan(instance)).toStdString();
            const auto& entity = compiledExpressions.at(core);
            const auto formal = ce::getArgs(entity.signature);
            const auto actual = ce::getArgs(instance);
            assert(formal.size() == actual.size());
            std::map<std::string, std::string> substitution;
            for (std::size_t i = 0; i < formal.size(); ++i) substitution[formal[i]] = actual[i];
            if (entity.category == "and") {
                for (const auto& element : entity.elements)
                    self(ce::replaceKeysInString(element, substitution), self);
            } else if (entity.category == "implication" && entity.elements.size() == 2) {
                if (isContrapositionTypingExclusion(entity)) return;
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
                std::string compiledHead = compileCoreExpressionMapCore(
                    head, implCounter, existenceCounter, andCounter, orCounter, variableCounter);
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
                const std::string sourceText = reconstructImplicationFullBind(
                    {sources[first].premise}, compiledHead);
                const std::string reverseText = reconstructImplicationFullBind(
                    {negate(compiledHead)}, negate(sources[first].premise));
                ContrapositionTemplate record;
                for (auto index : selected) {
                    record.sources.push_back(sources[index].compact);
                    covered[index] = true;
                }
                record.sourceRule = selected.size() == 1 ? record.sources.front()
                    : prefixArgumentsWithU(compileImplicationToCompact(sourceText));
                record.reverseRule = prefixArgumentsWithU(compileImplicationToCompact(reverseText));
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
        auto& target = compiledExpressions.at(name);
        target.contrapositives = std::move(prepared);
        target.contrapositionPrepared = true;
    }
    // A grouped source is a real compact carrier. Its own prepared reversal
    // reproduces the rule pair after mail, canonical rewriting or reload.
    for (const auto& name : groupedCarriers) {
        const LogicalEntity group = compiledExpressions.at(name);
        if (!group.contrapositives.empty()) continue;
        assert(group.category == "implication" && group.elements.size() == 2);
        std::map<std::string, std::string> renaming;
        for (const auto& argument : ce::getArgs(group.elements[0]))
            if (argument.rfind("u_", 0) != 0 && !renaming.count(argument))
                renaming[argument] = "contra_variable_" + std::to_string(renaming.size() + 1);
        const std::string premise = ce::replaceKeysInString(group.elements[0], renaming);
        const std::string head = ce::replaceKeysInString(group.elements[1], renaming);
        const std::string reverse = prefixArgumentsWithU(compileImplicationToCompact(
            reconstructImplicationFullBind({negate(head)}, negate(premise))));
        compiledExpressions.at(name).contrapositives.push_back(
            ContrapositionTemplate{{group.signature}, group.signature, reverse});
        compiledExpressions.at(name).contrapositionPrepared = true;
    }
    // A split source may have been prepared in an earlier completion seam.
    for (const auto& [name, entity] : compiledExpressions) {
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
    for (auto& [name, entity] : compiledExpressions)
        if (!originalNames.count(name)) {
            entity.contrapositionAuxiliary = true;
            entity.contrapositionPrepared = true;
        }
    compileConjunctionRefutations();
    compileSupportingImplications();
}

/// @brief Compile fixed-parameter refutations of conjunction members.
/// @details Walk positive conjunction descendants in definition order and
/// prepare each distinct member's complement-to-whole-complement implication.
/// All owner parameters remain fixed, including parameters absent in a member.
/// Direct helpers are excluded from the guarded-support generator.
/// @return Void.
/// @invariant Defining elements are unchanged and no runtime fact is published.
/// @see compileContrapositionFamilies.
void ExpressionAnalyzer::compileConjunctionRefutations() {
    std::vector<std::string> owners;
    std::set<std::string> originalNames;
    for (const auto& [name, entity] : compiledExpressions) {
        originalNames.insert(name);
        if (entity.category == "and" && !entity.conjunctionRefutationsPrepared)
            owners.push_back(name);
    }
    for (const auto& name : owners) {
        const LogicalEntity owner = compiledExpressions.at(name);
        std::vector<std::pair<std::string, std::string>> prepared;
        std::set<std::string> seen;
        const auto collect = [&](const std::string& conjunction, const auto& self) -> void {
            const auto& entity = compiledExpressions.at(
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
                const std::string rule = reconstructImplicationFullBind(
                    {negate(member)}, negate(owner.signature));
                const std::string compact = prefixArgumentsWithU(compileImplicationToCompact(rule));
                prepared.emplace_back(member, compact);
                const auto* child = compiledEntity(extractExpressionUniversalSpan(StrSpan(member)));
                if (member.front() == '(' && child != nullptr && child->category == "and")
                    self(member, self);
            }
        };
        collect(owner.signature, collect);
        auto& destination = compiledExpressions.at(name);
        destination.conjunctionRefutations = std::move(prepared);
        destination.conjunctionRefutationsPrepared = true;
    }
    for (auto& [name, entity] : compiledExpressions) {
        if (!originalNames.count(name)) {
            entity.contrapositionAuxiliary = true;
            entity.contrapositionPrepared = true;
        }
        for (const auto& record : entity.conjunctionRefutations) {
            auto& helper = compiledExpressions.at(
                extractExpressionUniversalSpan(StrSpan(record.second)).toStdString());
            helper.supportingImplicationAuxiliary = true;
        }
    }
}

/// @brief Compile one guarded supporting implication per immediate AND part.
/// @details Only ordinary implications and prepared contrapositives participate.
/// A helper complements one part and keeps every other signed part unchanged.
/// Parts remain opaque; helper entities cannot seed either auxiliary generator.
/// Unary owners receive direct compositions; their intermediate helpers remain
/// proof templates only. Non-unary owners retain the guarded helper rules.
/// @return Void.
/// @invariant All non-fixed conclusion arguments occur in the premise chain.
/// @see collectContrapositionProducts.
void ExpressionAnalyzer::compileSupportingImplications() {
    std::vector<std::string> owners;
    std::set<std::string> existing;
    for (const auto& [name, entity] : compiledExpressions) {
        existing.insert(name);
        if (entity.category == "implication" && !entity.supportingImplicationsPrepared
            && !entity.supportingImplicationAuxiliary) owners.push_back(name);
    }
    for (const auto& name : owners) {
        // Compiler reuse can make a previously registered compact a helper.
        if (compiledExpressions.at(name).supportingImplicationAuxiliary) continue;
        const LogicalEntity owner = compiledExpressions.at(name);
        assert(owner.elements.size() >= 2);
        const bool compose = owner.elements.size() == 2;
        std::vector<std::string> prepared, composed;
        std::set<std::string> seen;
        for (std::size_t p = 0; p + 1 < owner.elements.size(); ++p) {
            const std::string& premise = owner.elements[p];
            if (premise.empty() || premise.front() != '!') continue;
            const auto* definition = compiledEntity(extractExpressionUniversalSpan(StrSpan(premise)));
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
                chain[selected] = negate(chain[selected]);
                const std::string compact = prefixArgumentsWithU(compileImplicationToCompact(
                    reconstructImplicationFullBind(chain, helperHead)));
                if (seen.insert(compact).second) {
                    prepared.push_back(compact);
                    if (compose) {
                        for (const auto& argument : ce::getArgs(owner.elements.back()))
                            assert(argument.rfind("u_", 0) == 0 || covered.count(argument) != 0);
                        const std::string finalHead = ce::replaceKeysInString(owner.elements.back(), boundNames);
                        const std::string direct = prefixArgumentsWithU(compileImplicationToCompact(
                            reconstructImplicationFullBind(chain, finalHead)));
                        composed.push_back(direct);
                        auto& directRule = compiledExpressions.at(
                            extractExpressionUniversalSpan(StrSpan(direct)).toStdString());
                        directRule.supportingImplicationAuxiliary = true;
                        directRule.contrapositionAuxiliary = true;
                        directRule.contrapositionPrepared = true;
                    }
                }
                auto& helper = compiledExpressions.at(extractExpressionUniversalSpan(StrSpan(compact)).toStdString());
                helper.supportingImplicationAuxiliary = true;
                helper.contrapositionAuxiliary = true;
                helper.contrapositionPrepared = true;
            }
        }
        auto& destination = compiledExpressions.at(name);
        assert(composed.empty() || composed.size() == prepared.size());
        destination.supportingImplications = std::move(prepared);
        destination.composedSupportingImplications = std::move(composed);
        destination.supportingImplicationsPrepared = true;
    }
    for (auto& [name, entity] : compiledExpressions) if (!existing.count(name)) {
        entity.supportingImplicationAuxiliary = true;
        entity.supportingImplicationsPrepared = true;
        entity.contrapositionAuxiliary = true;
        entity.contrapositionPrepared = true;
    }
}

/// @brief Publish prepared auxiliary compacts through ordinary disintegration.
/// @details Only template substitution and checked provenance occur here. Each
/// contrapositive is a registered compact whose normal expansion installs its
/// own rule. Collected traversal markers break reciprocal activation cycles.
/// @param instance Fixed-argument source-family instance.
/// @param ownerKey Current collected statement key.
/// @param memoryBlock Resident target logic block.
/// @param collected Scratch products of this disintegration traversal.
/// @param validityName Exact source scope.
/// @param trackHistoryLocal Whether publication history is enabled.
/// @param iteration Source disintegration generation.
/// @return Void.
/// @invariant No registry mint, grouping or contraposition construction occurs here.
void ExpressionAnalyzer::collectContrapositionProducts(StrSpan instance,
    StrSpan ownerKey, Memory& memoryBlock, CollectedArena& collected,
    StrSpan validityName, bool trackHistoryLocal, int iteration) {
    const LogicalEntity* owner = compiledEntity(extractExpressionUniversalSpan(instance));
    assert(owner != nullptr);
    if (owner->contrapositives.empty() && owner->supportingImplications.empty()) return;
    const unsigned stringSlot = g_currentCoreId >= 0 ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    const unsigned containerSlot = g_currentCoreId >= 0 ? static_cast<unsigned>(g_currentCoreId)
        : genScratchArenas().slotCount() - 1;
    ScratchArena& arena = scratchArenas().forSlot(stringSlot);
    ScratchArena& containers = genScratchArenas().forSlot(containerSlot);
    ScratchScope frame(arena);
    StrSpan formal[ExecutionParameters::MAX_ARITY], actual[ExecutionParameters::MAX_ARITY];
    const int32_t count = getArgsSpans(StrSpan(owner->signature), formal, ExecutionParameters::MAX_ARITY);
    assert(count == getArgsSpans(instance, actual, ExecutionParameters::MAX_ARITY));
    StrReplacement ownerArguments[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < count; ++i) ownerArguments[i] = {formal[i], actual[i]};
    const auto instantiate = [&](const std::string& compactTemplate,
                                  ScratchString& compact, ScratchString& expanded) {
        compact = replaceKeysScratch(arena, StrSpan(compactTemplate), ownerArguments, count);
        const LogicalEntity* entity = compiledEntity(extractExpressionUniversalSpan(StrSpan(compact)));
        assert(entity && entity->category == "implication" && entity->elements.size() >= 2
            && entity->elements.size() <= ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
        StrSpan signatureArguments[ExecutionParameters::MAX_ARITY], instanceArguments[ExecutionParameters::MAX_ARITY];
        const int32_t arity = getArgsSpans(StrSpan(entity->signature), signatureArguments, ExecutionParameters::MAX_ARITY);
        assert(arity == getArgsSpans(StrSpan(compact), instanceArguments, ExecutionParameters::MAX_ARITY));
        StrReplacement replacements[ExecutionParameters::MAX_ARITY];
        for (int32_t i = 0; i < arity; ++i) replacements[i] = {signatureArguments[i], instanceArguments[i]};
        StrSpan elements[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
        const int32_t elementCount = static_cast<int32_t>(entity->elements.size());
        for (int32_t i = 0; i < elementCount; ++i)
            elements[i] = replaceKeysScratch(arena, StrSpan(entity->elements[i]), replacements, arity);
        expanded = reconstructImplicationFullBindScratch(arena, elements, elementCount - 1,
            elements[elementCount - 1]);
    };
    const auto started = [&](StrSpan key) {
        const int32_t id = collected.strings.lookup(key);
        if (id == 0) return false;
        for (int32_t i = 0; i < collected.recs.size(); ++i)
            if (collected.recs[i].keyId == id && collected.recs[i].kind == 2) return true;
        return false;
    };
    const int cap = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr
        : parameters.max_origin_per_expr;
    const auto recordHistory = [&](StrSpan expression, OriginTag tag,
                                   const OriginDep* dependencies, int32_t n) {
        if (!trackHistoryLocal || !parameters.trackHistory) return;
        addOriginEncoded(memoryBlock.exprOriginMap, memoryBlock.originInterner,
            expression, validityName, tag, dependencies, n, cap);
        memoryBlock.addMailOutOrigin(expression, validityName, tag, dependencies, n, cap);
    };
    const auto publish = [&](StrSpan compact, StrSpan clean, StrSpan expanded) {
        const OriginDep source{expanded, validityName};
        recordHistory(clean, OriginTag::compilation, &source, 1);
        collected.insertChild(ownerKey, clean);
        WorkInstruction instruction(&containers);
        NewVarStore witnesses(&containers);
        prepareIntegrationCore(compact, instruction, memoryBlock, clean, true);
        disintegrateExprCore2(compact, instruction, memoryBlock, iteration,
            collected, witnesses, validityName, trackHistoryLocal, false,
            nullptr, 0, false);
        assert(witnesses.varCount() == 0 && "auxiliary implication expansion minted a witness");
    };
    if (!owner->supportingImplications.empty()) {
        ScratchString sourceCompact, sourceExpanded;
        instantiate(owner->signature, sourceCompact, sourceExpanded);
        const OriginDep ownerDependency{ownerKey, validityName};
        recordHistory(StrSpan(sourceExpanded), OriginTag::expansion, &ownerDependency, 1);
        const bool composed = !owner->composedSupportingImplications.empty();
        assert(!composed || owner->composedSupportingImplications.size() == owner->supportingImplications.size());
        for (std::size_t i = 0; i < owner->supportingImplications.size(); ++i) {
            ScratchScope helperFrame(arena);
            ScratchString compact, expanded;
            instantiate(composed ? owner->composedSupportingImplications[i]
                : owner->supportingImplications[i], compact, expanded);
            const ScratchString carrier = removeUPrefixScratch(arena, StrSpan(compact));
            if (started(StrSpan(carrier))) continue;
            const OriginDep source{StrSpan(sourceExpanded), validityName};
            if (composed) {
                ScratchString helperCompact, helperExpanded;
                instantiate(owner->supportingImplications[i], helperCompact, helperExpanded);
                recordHistory(StrSpan(helperExpanded), OriginTag::expansion, &source, 1);
                const OriginDep dependencies[2] = {
                    {StrSpan(helperExpanded), validityName}, {StrSpan(sourceExpanded), validityName}};
                recordHistory(StrSpan(expanded), OriginTag::implicationComposition, dependencies, 2);
            } else {
                recordHistory(StrSpan(expanded), OriginTag::expansion, &source, 1);
            }
            publish(StrSpan(compact), StrSpan(carrier), StrSpan(expanded));
        }
    }
    for (const auto& record : owner->contrapositives) {
        ScratchScope recordFrame(arena);
        assert(!record.sources.empty());
        ScratchString sourceCompact, sourceHistory, reverseCompact, reverseHistory;
        instantiate(record.sourceRule, sourceCompact, sourceHistory);
        instantiate(record.reverseRule, reverseCompact, reverseHistory);
        const ScratchString sourceCarrier = removeUPrefixScratch(arena, StrSpan(sourceCompact));
        const ScratchString reverseCarrier = removeUPrefixScratch(arena, StrSpan(reverseCompact));
        if (record.sources.size() > 1 && !started(StrSpan(sourceCarrier))) {
            assert(record.sources.size() <= ExecutionParameters::kMaxOriginDeps);
            OriginDep dependencies[ExecutionParameters::kMaxOriginDeps];
            for (std::size_t i = 0; i < record.sources.size(); ++i) {
                ScratchString compact, history;
                instantiate(record.sources[i], compact, history);
                dependencies[i] = {StrSpan(history), validityName};
            }
            recordHistory(StrSpan(sourceHistory), OriginTag::implicationGrouping,
                dependencies, static_cast<int32_t>(record.sources.size()));
            publish(StrSpan(sourceCompact), StrSpan(sourceCarrier), StrSpan(sourceHistory));
        }
        if (started(StrSpan(reverseCarrier))) continue;
        const OriginDep source{StrSpan(sourceHistory), validityName};
        recordHistory(StrSpan(reverseHistory), OriginTag::contraposition, &source, 1);
        publish(StrSpan(reverseCompact), StrSpan(reverseCarrier), StrSpan(reverseHistory));
    }
}

/// @brief Publish prepared refutation compacts with exact structural history.
/// @details Every template is instantiated without changing the shared registry.
/// Structural expansion, compilation and ordinary rule expansion remain distinct.
/// @param mb Owning resident block.
/// @param conjunction Fully fixed conjunction instance.
/// @param scope Exact deepest supporting scope.
/// @param dependencies Sole encountered OR citation.
/// @param dependencyCount Exactly one supporting citation.
/// @param levels Sorted nonempty supporting levels.
/// @param levelCount Number of levels.
/// @param output Current internal-mail output channel.
/// @return Void.
void ExpressionAnalyzer::publishConjunctionRefutation(Memory& mb, StrSpan conjunction, NameId scope,
    const OriginDep* dependencies, int32_t dependencyCount, const int* levels, int32_t levelCount, ColdMail& output) {
    assert(dependencyCount == 1 && levelCount > 0);
    const auto* entity = compiledEntity(extractExpressionUniversalSpan(conjunction));
    assert(entity != nullptr && entity->category == "and" && entity->conjunctionRefutationsPrepared);
    const unsigned slot = g_currentCoreId >= 0 ? static_cast<unsigned>(g_currentCoreId) : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    const StrSpan scopeView = mb.nameMap.decodeView(scope);
    const ScratchString validity = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
    const ScratchString fixed = prefixArgumentsWithUScratch(strings, conjunction);
    StrSpan formal[ExecutionParameters::MAX_ARITY], actual[ExecutionParameters::MAX_ARITY];
    const int32_t arity = getArgsSpans(StrSpan(entity->signature), formal, ExecutionParameters::MAX_ARITY);
    assert(getArgsSpans(StrSpan(fixed), actual, ExecutionParameters::MAX_ARITY) == arity);
    StrReplacement mapping[ExecutionParameters::MAX_ARITY];
    for (int32_t i = 0; i < arity; ++i) mapping[i] = {formal[i], actual[i]};
    const int cap = parameters.compressor_mode ? parameters.compressor_max_origins_per_expr : parameters.max_origin_per_expr;
    for (const auto& row : entity->conjunctionRefutations) {
        const ScratchString compactU = replaceKeysScratch(strings, StrSpan(row.second), mapping, arity);
        const ScratchString compact = removeUPrefixScratch(strings, StrSpan(compactU));
        const auto* rule = compiledEntity(extractExpressionUniversalSpan(StrSpan(compactU)));
        assert(rule != nullptr && rule->category == "implication" && rule->elements.size() == 2);
        StrSpan ruleFormal[ExecutionParameters::MAX_ARITY], ruleActual[ExecutionParameters::MAX_ARITY];
        const int32_t ruleArity = getArgsSpans(StrSpan(rule->signature), ruleFormal, ExecutionParameters::MAX_ARITY);
        assert(getArgsSpans(StrSpan(compactU), ruleActual, ExecutionParameters::MAX_ARITY) == ruleArity);
        StrReplacement substitutions[ExecutionParameters::MAX_ARITY];
        for (int32_t i = 0; i < ruleArity; ++i) substitutions[i] = {ruleFormal[i], ruleActual[i]};
        const ScratchString premise = replaceKeysScratch(strings, StrSpan(rule->elements[0]), substitutions, ruleArity);
        const ScratchString head = replaceKeysScratch(strings, StrSpan(rule->elements[1]), substitutions, ruleArity);
        const StrSpan elements[2] = {StrSpan(premise), StrSpan(head)};
        const ScratchString expanded = expandSignature(StrSpan("implication", 11), StrSpan(compactU), elements, 2, strings);
        if (parameters.trackHistory && !ceFilteringActive) {
            addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(expanded), StrSpan(validity),
                OriginTag::expansion, dependencies, dependencyCount, cap);
            mb.addMailOutOrigin(StrSpan(expanded), StrSpan(validity), OriginTag::expansion, dependencies, dependencyCount, cap);
            const OriginDep dependency{StrSpan(expanded), StrSpan(validity)};
            addOriginEncoded(mb.exprOriginMap, mb.originInterner, StrSpan(compact), StrSpan(validity),
                OriginTag::compilation, &dependency, 1, cap);
            mb.addMailOutOrigin(StrSpan(compact), StrSpan(validity), OriginTag::compilation, &dependency, 1, cap);
            addInternalMailOrigin(output, mb.originInterner, StrSpan(compact), StrSpan(validity),
                OriginTag::compilation, &dependency, 1, cap);
        }
        insertInternalStatement(output, mb.nameMap, StrSpan(compact), StrSpan(validity), levels, levelCount);
        setInternalDisintegrationSignal(output, mb.nameMap, StrSpan(compact), StrSpan(validity), false, false, 0);
        mb.mutatedThisBurst = true;
    }
}

/// @brief Publish fixed direct refutations for an admitted OR's AND alternatives.
/// @details The known OR fixes every argument locally. No reverse-rule search,
/// descendant replay or equality index is created. Ordinary compact lifecycle
/// handles later canonical changes and scope cleanup.
/// @param mb Owning block with the admitted OR and its supporting levels.
/// @param statement Exact positive OR source.
/// @param output Current processing phase's output mail.
/// @return Void.
void ExpressionAnalyzer::activateDirectOrRefutations(Memory& mb,
    const IntEncodedExpr& statement, ColdMail& output) {
    if (parameters.ban_disintegration || mb.phase3IgnoresScope(statement.validityId)) return;
    const unsigned slot = g_currentCoreId >= 0 ? static_cast<unsigned>(g_currentCoreId)
        : scratchArenas().slotCount() - 1;
    ScratchArena& strings = scratchArenas().forSlot(slot);
    ScratchScope frame(strings);
    const StrSpan textView = mb.nameMap.decodeView(statement.originalId);
    const ScratchString text = ScratchString::copyFrom(strings, textView.ptr, textView.len);
    const StrSpan scopeView = mb.nameMap.decodeView(statement.validityId);
    const ScratchString scope = ScratchString::copyFrom(strings, scopeView.ptr, scopeView.len);
    const auto* entity = compiledEntity(extractExpressionUniversalSpan(StrSpan(text)));
    assert(!statement.negation && entity && entity->category == "or");
    const int32_t levelRow = mb.intStatementLevelsMap.lookup(
        packStatementKey(statement.originalId, statement.validityId));
    assert(levelRow != 0);
    const int32_t levelCount = mb.intStatementLevelsMap.runLen(levelRow);
    assert(levelCount > 0 && levelCount <= 256);
    int levels[256];
    for (int32_t i = 0; i < levelCount; ++i) levels[i] = mb.intStatementLevelsMap.valueAt(levelRow, i);
    StrSpan leaves[ExecutionParameters::MAX_INSTRUCTION_ELEMENTS];
    const int32_t count = flattenRegistryOrLeavesScratch(*entity, StrSpan(text), leaves,
        ExecutionParameters::MAX_INSTRUCTION_ELEMENTS, strings);
    const OriginDep source{StrSpan(text), StrSpan(scope)};
    for (int32_t i = 0; i < count; ++i) {
        const auto* part = compiledEntity(extractExpressionUniversalSpan(leaves[i]));
        if (leaves[i].ptr[0] != '(' || part == nullptr || part->category != "and") continue;
        publishConjunctionRefutation(mb, leaves[i], statement.validityId, &source, 1,
            levels, levelCount, output);
    }
}

} // namespace gl
