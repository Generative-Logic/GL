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
#include "compilation_reuse.hpp"
#include "compiled_definition_registry.hpp"

namespace gl {

/// @brief Bind the definition stores used by reuse operations.
/// @details References remain valid for this compiler instance lifetime.
/// @param definitions Definition registry supplying the private stores.
/// @param permutations Prepared ordered premise permutations.
/// @return Constructed reuse component.
CompilationReuse::CompilationReuse(CompiledDefinitionRegistry& definitions,
    std::map<int, std::vector<std::vector<int>>>& permutations)
    : compiledExpressions(definitions.compiledExpressions), coreExpressionMap(definitions.coreExpressionMap),
      allPermutationsAna(permutations) {}

/// @brief Retain a public OR's argument mapping for an ordered body.
/// @details Replaces the same body's cached mapping in its existing order.
/// @param elements Ordered normalized OR elements.
/// @param name Retained public predicate.
/// @param positions Argument positions in the retained predicate.
/// @return None.
void CompilationReuse::rememberOrReuse(const std::vector<std::string>& elements,
    const std::string& name, const std::vector<int>& positions) {
    compiledOrReuse[elements] = {name, positions};
}

/// @brief Restore one generated-name replacement from compilation input.
/// @details A repeated replacement must agree exactly with its stored row.
/// @param name Generated name to replace.
/// @param replacement Retained name and argument permutation.
/// @return None.
void CompilationReuse::restoreReplacement(const std::string& name,
    const CompilationReplacement& replacement) {
    const auto inserted = compilationReplacements.emplace(name, replacement);
    assert(inserted.second || (inserted.first->second.target == replacement.target
        && inserted.first->second.arguments == replacement.arguments));
}

/// @brief Clear reuse indexes and replacement metadata for a fresh compiler input.
/// @details Definition stores and the public-name phase state are unchanged.
/// @return None.
void CompilationReuse::clear() {
    repetitionExclusionMap.clear(); namedRepetitionExclusionMap.clear();
    compiledOrReuse.clear(); namedCompilationOrder.clear();
    namedCompilationNames.clear(); compilationReplacements.clear();
}
/// @brief Replace the complete value state with a recorded one.
///
/// @details
/// Every reuse index, the replacement metadata and the public-name phase flag
/// become copies of @p recorded; the bound definition stores are not touched.
/// The caller installs the matching registry, so the restored indexes name the
/// entities they were recorded with.
///
/// @param recorded State read through `state` from a component whose registry
///        equals this component's registry.
/// @return None.
/// @see state
void CompilationReuse::restoreState(const CompilationReuseState& recorded) {
    static_cast<CompilationReuseState&>(*this) = recorded;
}
/// @brief Reuse or register a structurally equivalent compiled definition.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param encodedTail Ordered premise expressions.
/// @param encodedHead Ordered head expressions.
/// @param newCoreExpressionName Candidate generated or public name.
/// @param category Logical definition category.
/// @param force Whether this is a forced public registration.
/// @param defaultSignature Public signature retained for forced registration.
/// @return Retained compact expression.
std::string CompilationReuse::excludeRepetitions(const std::vector<EncodedExpression>& encodedTail, const std::vector<EncodedExpression>& encodedHead, const std::string& newCoreExpressionName, const std::string& category, const bool force, const std::string defaultSignature) {
            std::string result;

			std::vector<EncodedExpression> encodedList = encodedTail;
			encodedList.insert(encodedList.end(), encodedHead.begin(), encodedHead.end());
            std::pair<NormalizedKey, std::map<std::string, std::string>> pr2 = gl::CompilationNormalizer::makeNormalizedEncodedKey(encodedList, 2);
            std::vector<std::string> splitNK = splitNormalizedKey(pr2.first.data);

            // Every definition, public or numbered, is keyed by its canonical
            // body (canonicalReuseKey): one key per definition, found whatever
            // the premise order and variable names of the body at hand. A body
            // without premises is neither looked up nor registered.
            const bool keyed = !encodedTail.empty();
            const auto canonical = keyed ? canonicalReuseKey(encodedTail, encodedHead)
                : std::pair<std::vector<std::string>, std::map<std::string, std::string>>{};

            if (force)
            {
                std::map<std::string, std::string> replacementMap;

                std::vector<std::string> argsDefaultSignature = ce::getArgs(defaultSignature);
                // Pass .begin() and .end() iterators
                assert(std::set<std::string>(argsDefaultSignature.begin(), argsDefaultSignature.end()).size() == argsDefaultSignature.size());

				for (int ind = 1; ind <= static_cast<int>(argsDefaultSignature.size()); ind++)
                {
                    std::string value = "u_" + std::to_string(ind);

                    assert(!startsWithSpan(StrSpan(argsDefaultSignature[static_cast<std::size_t>(ind) - 1]), "u_", 2));
                    std::string key = "u_" + argsDefaultSignature[static_cast<std::size_t>(ind) - 1];
                    replacementMap[key] = value;
                }
                std::map<std::string, std::string> revUnchMap = gl::CompilationNormalizer::createReverseUnchangeableMap(replacementMap, pr2.second);

                std::vector<std::string> renSplitNK;
                for (const auto& elem : splitNK)
                {
                    renSplitNK.push_back(ce::replaceKeysInString(elem, revUnchMap));
                }

                splitNK = renSplitNK;
                pr2.first.data = splitNK;
                pr2.second = replacementMap;
            }

            // Category-keyed lookup: see repetitionExclusionMap declaration. A
            // forced public registration never reuses, so it looks nothing up.
            const CompilationReuseMap::const_iterator it = (force || !keyed) ? repetitionExclusionMap.end()
                : repetitionExclusionMap.find(std::make_pair(canonical.first, category));
            if (it != repetitionExclusionMap.end())
            {
                const std::tuple<std::map<std::string, std::string>,
                    std::string,
                    std::vector<std::string>>&tpl = it->second;
                const std::map<std::string, std::string>& namePermutation = std::get<0>(tpl);
                const std::string& coreExpression = std::get<1>(tpl);
				const std::vector<std::string>& mainNormalizedDefinition = std::get<2>(tpl);

				std::map<std::string, std::string> combinedMap = gl::CompilationNormalizer::combineUnchangeablePaths(canonical.second, namePermutation);
				std::string args = gl::CompilationNormalizer::createArgList(combinedMap);

				std::string expression = "(" + coreExpression;
				expression += args;
				expression += ")";
				result = expression;
            }
            else
            {
				int numUnchArgs = 0;

                if (keyed)
                {
                    // One entry: the canonical key, its slots mapped to the
                    // slots of the definition's own signature (the stored order,
                    // or the declared order of a public signature).
                    std::map<std::string, std::string> reverseUnchMap = gl::CompilationNormalizer::createReverseUnchangeableMap(pr2.second, canonical.second);
                    numUnchArgs = static_cast<int>(reverseUnchMap.size());
                    const std::string coreName = force
                        ? extractExpressionUniversalSpan(StrSpan(defaultSignature)).toStdString()
                        : newCoreExpressionName;
                    registerCompilationReuse(std::make_pair(canonical.first, category),
                        std::make_tuple(reverseUnchMap, coreName, splitNK), force);
                }

                if (!force)
                {
                    std::string signature = "(" + newCoreExpressionName + "[";
                    for (int index = 1; index <= numUnchArgs; index++)
                    {
                        signature += "u_" + std::to_string(index) + ",";
                    }
                    if (numUnchArgs > 0)
                    {
                        signature.pop_back(); // remove the trailing comma the loop appended
                    }
                    // numUnchArgs == 0: no comma was appended; popping would eat
                    // the '[' and yield the malformed "(nameN])". A zero-free-arg
                    // compile (a fully-bound theorem) must yield "(nameN[])".
                    signature += "])";

                    {
                        std::string preservedDS;
                        auto itPrev = compiledExpressions.find(newCoreExpressionName);
                        if (itPrev != compiledExpressions.end())
                            preservedDS = itPrev->second.definedSet;
                        compiledExpressions[newCoreExpressionName] = LogicalEntity(category, splitNK, signature, numUnchArgs, preservedDS);
                    }

                    std::string args = gl::CompilationNormalizer::createArgList(pr2.second);

                    std::string expression = "(" + newCoreExpressionName;
                    expression += args;
                    expression += ")";
                    result = expression;
                }
                else
                {
                    std::string coreExpressionName = extractExpressionUniversalSpan(StrSpan(defaultSignature)).toStdString();
                    std::string signature = "(" + coreExpressionName + "[";
                    for (int index = 1; index <= numUnchArgs; index++)
                    {
                        signature += "u_" + std::to_string(index) + ",";
                    }
                    if (numUnchArgs > 0)
                    {
                        signature.pop_back(); // remove the trailing comma the loop appended
                    }
                    // numUnchArgs == 0: no comma was appended; popping would eat
                    // the '[' and yield the malformed "(nameN])". A zero-free-arg
                    // compile (a fully-bound theorem) must yield "(nameN[])".
                    signature += "])";

                    {
                        std::string preservedDS;
                        auto itPrev = compiledExpressions.find(coreExpressionName);
                        if (itPrev != compiledExpressions.end())
                            preservedDS = itPrev->second.definedSet;
                        compiledExpressions[coreExpressionName] = LogicalEntity(category, splitNK, signature, numUnchArgs, preservedDS);
                    }
                    result = defaultSignature;
                }


            }

			return result;
        }

/// @brief Find a compiled OR by its ordered defining elements.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param elems Ordered defining elements.
/// @param elemN Number of elements.
/// @param argumentPositions Optional destination for retained argument positions.
/// @param capacity Capacity of the optional destination.
/// @return Stable retained name, or null for the defined no-match case.
const std::string* CompilationReuse::compiledOrByElements(const StrSpan* elems, int32_t elemN, int* argumentPositions, int capacity) const {
            for (const auto& [elements, retained] : compiledOrReuse) {
                if (static_cast<int32_t>(elements.size()) != elemN) continue;
                bool same = true;
                for (int32_t i = 0; i < elemN && same; ++i)
                    same = equalSpans(StrSpan(elements[i]), elems[i]);
                if (!same) continue;
                assert(argumentPositions == nullptr || capacity >= static_cast<int>(retained.second.size()));
                for (std::size_t i = 0; i < retained.second.size(); ++i) {
                    assert(argumentPositions != nullptr || retained.second[i] == static_cast<int>(i + 1));
                    if (argumentPositions != nullptr) argumentPositions[i] = retained.second[i];
                }
                return &retained.first;
            }
            for (const auto& [name, le] : compiledExpressions) {
                if (le.category != "or") continue;
                if (static_cast<int32_t>(le.elements.size()) != elemN) continue;
                bool eq = true;
                for (int32_t i = 0; i < elemN && eq; ++i) {
                    eq = equalSpans(StrSpan(le.elements[i]), elems[i]);
                }
                if (eq) {
                    if (argumentPositions != nullptr) {
                        assert(capacity >= le.arity);
                        for (int i = 0; i < le.arity; ++i) argumentPositions[i] = i + 1;
                    }
                    return &name;
                }
            }
            return nullptr;
        }

/// @brief The repetition-map key of a body: its canonical order, normalized.
///
/// @details
/// Puts the premises in canonical order (`CompilationNormalizer::canonicalCombination`),
/// appends the head, normalizes the list in mode 2 and splits it into element
/// strings. Two bodies that differ only in premise order and variable names
/// get the same key. The mapping names, for every variable of the body as
/// given, its normalized spelling in the key: `u_<k>` for the k-th distinct
/// `u_` variable of the canonical order, the k-th bound variable as `k`.
///
/// @param tail Premises in any order.
/// @param head Elements kept after the premises (an implication's conclusion).
/// @return The key elements and the variable mapping.
/// @see CompilationNormalizer::canonicalCombination, excludeRepetitions,
///      compilationReuseEntries
std::pair<std::vector<std::string>, std::map<std::string, std::string>> CompilationReuse::canonicalReuseKey(
    const std::vector<EncodedExpression>& tail, const std::vector<EncodedExpression>& head) {
    std::vector<EncodedExpression> list = gl::CompilationNormalizer::canonicalCombination(tail, head);
    list.insert(list.end(), head.begin(), head.end());
    const auto normalized = gl::CompilationNormalizer::makeNormalizedEncodedKey(list, 2);
    return {splitCompilationKey(list, normalized.first.data), normalized.second};
}

/// @brief Split a normalized key using the source expression arities.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param expressions Source expressions carrying their arities.
/// @param normalized Normalized key tokens in source order.
/// @return Ordered normalized expressions.
std::vector<std::string> CompilationReuse::splitCompilationKey(const std::vector<EncodedExpression>& expressions, const std::vector<std::string>& normalized) {
    std::vector<std::string> result;
    std::size_t position = 0;
    for (const auto& expression : expressions) {
        const auto arity = expression.arguments.size();
        assert(position + 2 + arity <= normalized.size());
        assert(normalized[position] == expression.name);
        std::string element = (expression.negation ? "!(" : "(") + normalized[position] + '[';
        assert(normalized[position + 1] == (expression.negation ? "True" : "False"));
        position += 2;
        for (std::size_t index = 0; index < arity; ++index) {
            if (index) element += ',';
            element += normalized[position++];
        }
        result.push_back(element + "])");
    }
    assert(position == normalized.size());
    return result;
}

/// @brief Register a reuse identity with existing public-name precedence.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param key Normalized body and logical category.
/// @param value Argument mapping, retained name and defining elements.
/// @param named Whether this is a public definition.
/// @return None.
void CompilationReuse::registerCompilationReuse(const CompilationReuseKey& key, const CompilationReuseValue& value, bool named) {
    const auto& name = std::get<1>(value);
    if (named && namedCompilationNames.insert(name).second) namedCompilationOrder.push_back(name);
    auto winner = namedRepetitionExclusionMap.find(key);
    if (named && winner == namedRepetitionExclusionMap.end()) {
        winner = namedRepetitionExclusionMap.emplace(key, value).first;
    }
    if (winner != namedRepetitionExclusionMap.end()) {
        const auto previous = repetitionExclusionMap.find(key);
        if (previous != repetitionExclusionMap.end()
            && !namedCompilationNames.count(std::get<1>(previous->second))) {
            retireNumberedCompilation(previous->second, winner->second);
        }
        if (!named && compiledExpressions.count(name)) retireNumberedCompilation(value, winner->second);
        repetitionExclusionMap[key] = winner->second;
    } else {
        repetitionExclusionMap[key] = value;
    }
}

/// @brief The reuse identity of a compiled definition.
///
/// @details One entry: the canonical key of the definition's elements
/// (`canonicalReuseKey`, an implication's last element as head) with the map
/// from its canonical slots to the definition's own signature slots.
/// @param name Registered definition to inspect.
/// @return The definition's single reuse identity.
std::vector<std::pair<CompilationReuseKey, CompilationReuseValue>> CompilationReuse::compilationReuseEntries(const std::string& name) {
    assert(compiledExpressions.count(name));
    const auto& entity = compiledExpressions.at(name);
    assert(entity.category == "and" || entity.category == "or"
           || entity.category == "existence" || entity.category == "implication");
    assert(!entity.elements.empty() && "logical entity must have elements");
    std::vector<EncodedExpression> tail;
    for (const auto& element : entity.elements) tail.emplace_back(element, "main");
    std::vector<EncodedExpression> head;
    if (entity.category == "implication") {
        head.push_back(tail.back());
        tail.pop_back();
    }
    std::map<std::string, std::string> signature;
    for (const auto& argument : ce::getArgs(entity.signature)) signature[argument] = argument;
    const auto canonical = canonicalReuseKey(tail, head);
    const auto mapping = gl::CompilationNormalizer::createReverseUnchangeableMap(signature, canonical.second);
    assert(mapping.size() == signature.size() && "every public signature slot must occur in its body");
    return {{CompilationReuseKey{canonical.first, entity.category},
             CompilationReuseValue{mapping, name, entity.elements}}};
}

/// @brief Restore the reuse identities of one loaded public definition.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param name Loaded public definition.
/// @return None.
void CompilationReuse::registerLoadedNamedCompilation(const std::string& name) {
    registerStoredCompilation(name, true);
}

/// @brief Register the normalized identities of an already stored definition.
/// @details Preserves the identity enumeration and public-name precedence order.
/// @param name Stored definition to register.
/// @param named Whether the definition carries a public name.
/// @return None.
void CompilationReuse::registerStoredCompilation(const std::string& name, bool named) {
    for (const auto& entry : compilationReuseEntries(name))
        registerCompilationReuse(entry.first, entry.second, named);
}

/// @brief Record the argument-correct replacement of a numbered definition.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param numbered Numbered definition being replaced.
/// @param named Retained public definition.
/// @return None.
void CompilationReuse::retireNumberedCompilation(const CompilationReuseValue& numbered, const CompilationReuseValue& named) {
    const std::string& oldName = std::get<1>(numbered);
    const std::string& newName = std::get<1>(named);
    if (oldName == newName || compilationReplacements.count(oldName)) return;
    assert(!namedCompilationNames.count(oldName) && namedCompilationNames.count(newName));
    assert(!compilationNamesFrozen && "named promotion after proof-state creation");
    const auto& oldMap = std::get<0>(numbered);
    const auto& newMap = std::get<0>(named);
    assert(oldMap.size() == newMap.size());
    CompilationReplacement replacement{newName, std::vector<int>(newMap.size(), 0)};
    for (const auto& [slot, newArgument] : newMap) {
        const auto oldArgument = oldMap.find(slot);
        assert(oldArgument != oldMap.end());
        assert(newArgument.rfind("u_", 0) == 0 && oldArgument->second.rfind("u_", 0) == 0);
        const int newIndex = std::stoi(newArgument.substr(2));
        const int oldIndex = std::stoi(oldArgument->second.substr(2));
        assert(newIndex > 0 && static_cast<std::size_t>(newIndex) <= replacement.arguments.size());
        replacement.arguments[static_cast<std::size_t>(newIndex) - 1] = oldIndex;
    }
    const std::set<int> positions(replacement.arguments.begin(), replacement.arguments.end());
    assert(positions.size() == replacement.arguments.size());
    assert(positions.empty() || (*positions.begin() == 1 && *positions.rbegin() == static_cast<int>(positions.size())));
    compilationReplacements.emplace(oldName, std::move(replacement));
}

/// @brief Rewrite and retire numbered definitions before proof consumers start.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @return None.
void CompilationReuse::finalizeCompilationReplacements() {
    compiledOrReuse.clear();
    // Compose inherited replacement chains before erasing any intermediate
    // definition or retargeting a cache entry to its retained destination.
    auto flattened = compilationReplacements;
    for (const auto& [name, replacement] : compilationReplacements) {
        std::string signature = '(' + name + '[';
        for (std::size_t i = 0; i < replacement.arguments.size(); ++i) {
            if (i) signature += ',';
            signature += std::to_string(i + 1);
        }
        signature += "])";
        const auto rewritten = rewriteCompilationReferences(signature, compilationReplacements);
        auto& destination = flattened.at(name);
        destination.target = extractExpressionUniversalSpan(StrSpan(rewritten)).toStdString();
        const auto arguments = ce::getArgs(rewritten);
        destination.arguments.clear();
        for (const auto& argument : arguments) destination.arguments.push_back(std::stoi(argument));
        assert(compiledExpressions.count(destination.target));
    }
    compilationReplacements = std::move(flattened);
    bool changed = true;
    while (changed) {
        const auto previousCount = compilationReplacements.size();
        std::set<std::pair<std::string, std::size_t>> namedShapes;
        for (const auto& [key, value] : namedRepetitionExclusionMap) {
            (void)value;
            namedShapes.emplace(key.second, key.first.size());
        }
        for (const auto& [name, entity] : compiledExpressions) {
            if (!isNumberedCompilationName(name, entity.category)
                || compilationReplacements.count(name)
                || !namedShapes.count({entity.category, entity.elements.size()})) continue;
            for (const auto& entry : compilationReuseEntries(name)) {
                const auto winner = namedRepetitionExclusionMap.find(entry.first);
                if (winner != namedRepetitionExclusionMap.end()) {
                    retireNumberedCompilation(entry.second, winner->second);
                    break;
                }
            }
        }
        changed = compilationReplacements.size() != previousCount;
        if (compilationReplacements.empty()) break;
        for (auto& [name, entity] : compiledExpressions) {
            for (auto& element : entity.elements) element = rewriteCompilationReferences(element, compilationReplacements);
            for (auto& implication : entity.implications) implication = rewriteCompilationReferences(implication, compilationReplacements);
            for (auto& record : entity.integrationImplications)
                record.compact = rewriteCompilationReferences(record.compact, compilationReplacements);
            for (auto& rule : entity.supportingImplications)
                rule = rewriteCompilationReferences(rule, compilationReplacements);
            for (auto& rule : entity.composedSupportingImplications)
                rule = rewriteCompilationReferences(rule, compilationReplacements);
            for (auto& rule : entity.conjunctionRefutations) {
                rule.first = rewriteCompilationReferences(rule.first, compilationReplacements);
                rule.second = rewriteCompilationReferences(rule.second, compilationReplacements);
            }
            for (auto& rule : entity.contrapositives) {
                for (auto& source : rule.sources)
                    source = rewriteCompilationReferences(source, compilationReplacements);
                rule.sourceRule = rewriteCompilationReferences(rule.sourceRule, compilationReplacements);
                rule.reverseRule = rewriteCompilationReferences(rule.reverseRule, compilationReplacements);
            }
            for (auto& binder : entity.shotBinders)
                binder = rewriteCompilationReferences(binder, compilationReplacements);
            for (auto& shot : entity.shots) {
                shot.universal = rewriteCompilationReferences(shot.universal, compilationReplacements);
                for (auto& premise : shot.premises)
                    premise = rewriteCompilationReferences(premise, compilationReplacements);
                shot.head = rewriteCompilationReferences(shot.head, compilationReplacements);
            }
        }
        for (const auto& [name, replacement] : compilationReplacements) {
            (void)replacement;
            changed = compiledExpressions.erase(name) != 0 || changed;
            coreExpressionMap.erase(name);
        }
        for (auto& [name, config] : coreExpressionMap) {
            config.signature = rewriteCompilationReferences(config.signature, compilationReplacements);
            if (std::holds_alternative<std::string>(config.definition)) {
                config.definition = rewriteCompilationReferences(std::get<std::string>(config.definition), compilationReplacements);
            }
        }
        CompilationReuseMap rewritten;
        for (const auto& [key, value] : repetitionExclusionMap) {
            std::vector<std::string> elements;
            bool affected = compilationReplacements.count(std::get<1>(value)) != 0;
            for (const auto& element : key.first) {
                elements.push_back(rewriteCompilationReferences(element, compilationReplacements));
                affected = affected || elements.back() != element;
            }
            if (!affected) {
                rewritten[key] = value;
                continue;
            }
            // A renamed element can move to another name group, so the
            // rewritten body is keyed canonically again.
            std::vector<EncodedExpression> tail;
            for (const auto& element : elements) tail.emplace_back(element, "main");
            std::vector<EncodedExpression> head;
            if (key.second == "implication") {
                head.push_back(tail.back());
                tail.pop_back();
            }
            const auto canonical = canonicalReuseKey(tail, head);
            auto mapping = gl::CompilationNormalizer::createReverseUnchangeableMap(std::get<0>(value), canonical.second);
            std::string name = std::get<1>(value);
            const auto replacement = compilationReplacements.find(name);
            if (replacement != compilationReplacements.end()) {
                const auto& permutation = replacement->second.arguments;
                for (auto& [slot, argument] : mapping) {
                    (void)slot;
                    const int oldIndex = std::stoi(argument.substr(2));
                    const auto found = std::find(permutation.begin(), permutation.end(), oldIndex);
                    assert(found != permutation.end());
                    argument = "u_" + std::to_string(std::distance(permutation.begin(), found) + 1);
                }
                name = replacement->second.target;
            }
            assert(compiledExpressions.count(name));
            rewritten[{canonical.first, key.second}] =
                CompilationReuseValue{mapping, name, compiledExpressions.at(name).elements};
        }
        repetitionExclusionMap = std::move(rewritten);
        namedRepetitionExclusionMap.clear();
        const auto order = namedCompilationOrder;
        for (const auto& name : order) registerLoadedNamedCompilation(name);
        changed = changed || compilationReplacements.size() != previousCount;
    }
    for (const auto& [name, entity] : compiledExpressions) {
        assert(!compilationReplacements.count(name));
        for (const auto& element : entity.elements)
            assert(rewriteCompilationReferences(element, compilationReplacements) == element);
        for (const auto& implication : entity.implications)
            assert(rewriteCompilationReferences(implication, compilationReplacements) == implication);
        for (const auto& record : entity.integrationImplications)
            assert(rewriteCompilationReferences(record.compact, compilationReplacements) == record.compact);
    }
}

/// @brief Reconstruct ordered expression strings from a normalized key.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param normalizedKey Tokens with operator, polarity and normalized arguments.
/// @return Ordered expression strings.
std::vector<std::string> CompilationReuse::splitNormalizedKey(const std::vector<std::string> normalizedKey) {
            std::vector<std::string> result;

            int position = 0;
            while (true)
            {
                const LogicalEntity& le = compiledExpressions[normalizedKey[position]];
                std::string coreExpression = normalizedKey[position];
                position++;

                std::string expression = "(" + coreExpression + "[";
                bool isNegation = normalizedKey[position] == "True";
                position++;

                if (isNegation)
                {
                    expression = "!" + expression;
                }

                int arity = le.arity;
                for (int index = 0; index < arity; index++)
                {
                    expression += normalizedKey[position] + ",";
                    position++;
                }
                expression.pop_back();
                expression += "])";
                result.push_back(expression);

                if (position == normalizedKey.size())
                {
                    break;
                }

            }

            return result;
        }

} // namespace gl
