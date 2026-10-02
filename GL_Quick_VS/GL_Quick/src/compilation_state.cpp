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
#include <numeric>

namespace gl {

/// @brief Mark an OR whose two digit parameters are interchangeable.
///
/// @details Uses definition-slot types, positive OR flattening, signed leaf
/// multiplicity and equality symmetry. Free parameter names never normalize
/// away, and non-equality predicates keep their argument order and binding.
/// @param orName Registered OR operator whose derived metadata is refreshed.
/// @return Void.
/// @invariant No compiled expression or proof spelling is rewritten.
/// @see canonicalOrProcessingForm; preMintReducedOrs.
void ExpressionAnalyzer::compileOrSymmetry(const std::string& orName) {
    auto& entity = compiledExpressions.at(orName);
    assert(entity.category == "or");
    entity.symmetricDigitSlots = {0, 0};
    if (!isNumberedCompilationName(orName, "or")) return;
    const auto parameters = ce::getArgs(entity.signature);
    std::set<std::string> digits;
    const auto collectTypes = [&](const std::string& instance, int depth,
                                  const auto& self) -> void {
        assert(depth < ExecutionParameters::MAX_INSTRUCTION_ELEMENTS);
        const StrSpan core = extractExpressionUniversalSpan(StrSpan(instance));
        const auto arguments = ce::getArgs(instance);
        const auto* config = coreConfig(core);
        bool allTyped = config != nullptr && !config->definitionSets.empty();
        for (std::size_t i = 0; i < arguments.size(); ++i) {
            const auto* type = config == nullptr ? nullptr
                : findDefinitionSet(config, static_cast<int>(i) + 1);
            allTyped = allTyped && type != nullptr;
            if (type != nullptr && type->first == "(1)"
                && std::find(parameters.begin(), parameters.end(), arguments[i]) != parameters.end())
                digits.insert(arguments[i]);
        }
        if (allTyped) return;
        const auto* child = compiledEntity(core);
        if (child == nullptr || child->category == "atomic") return;
        const auto formal = ce::getArgs(child->signature);
        assert(formal.size() == arguments.size());
        std::map<std::string, std::string> substitution;
        for (std::size_t i = 0; i < formal.size(); ++i)
            substitution[formal[i]] = arguments[i];
        for (const auto& element : child->elements)
            self(ce::replaceKeysInString(element, substitution), depth + 1, self);
    };
    collectTypes(entity.signature, 0, collectTypes);
    if (digits.size() != 2) return;
    int positions[2];
    int count = 0;
    for (std::size_t i = 0; i < parameters.size(); ++i)
        if (digits.count(parameters[i])) positions[count++] = static_cast<int>(i);
    assert(count == 2);
    std::vector<std::string> leaves;
    flattenRegistryOrLeaves(entity, leaves);
    const std::map<std::string, std::string> swap{
        {parameters[positions[0]], parameters[positions[1]]},
        {parameters[positions[1]], parameters[positions[0]]}};
    const auto normalized = [&](bool exchanged) {
        std::vector<std::string> result;
        for (const auto& leaf : leaves) {
            std::string value = exchanged ? ce::replaceKeysInString(leaf, swap) : leaf;
            if (equalSpans(extractExpressionUniversalSpan(StrSpan(value)), StrSpan("=", 1))) {
                auto arguments = ce::getArgs(value);
                assert(arguments.size() == 2);
                if (arguments[1] < arguments[0]) std::swap(arguments[0], arguments[1]);
                value = (value[0] == '!' ? "!(=[" : "(=[")
                    + arguments[0] + "," + arguments[1] + "])";
            }
            result.push_back(std::move(value));
        }
        std::sort(result.begin(), result.end());
        return result;
    };
    if (normalized(false) == normalized(true))
        entity.symmetricDigitSlots = {static_cast<int16_t>(positions[0] + 1),
                                      static_cast<int16_t>(positions[1] + 1)};
}

/// @brief Split a normalized compilation key using its source arities.
///
/// @details The encoded expressions make every boundary explicit, including
/// zero-arity atoms. No dependency needs to have been loaded or compiled yet.
/// @param expressions Encoded expressions in key order.
/// @param normalized Flat normalized key.
/// @return Normalized MPL elements with the original element order.
/// @invariant Every flat token is consumed exactly once.
std::vector<std::string> ExpressionAnalyzer::splitCompilationKey(
    const std::vector<EncodedExpression>& expressions,
    const std::vector<std::string>& normalized) {
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

/// @brief Distinguish generated operator names from public predicate names.
///
/// @details The generated spelling is category plus a nonempty decimal suffix.
/// @param name Operator name.
/// @param category Logical category.
/// @return Whether the name has the generated spelling.
bool isNumberedCompilationName(const std::string& name, const std::string& category) {
    return isNumberedCompilationName(StrSpan(name), StrSpan(category));
}

/// @brief Classify a compiled operator name without allocating string storage.
///
/// @details The generated spelling is one of the four compilation categories
/// followed by decimal digits. The spans are read only during this call.
/// @param name Operator name, without expression punctuation.
/// @param category Logical category.
/// @return Whether the name has the generated spelling.
/// @see isNumberedCompilationName(const std::string&, const std::string&).
bool isNumberedCompilationName(StrSpan name, StrSpan category) {
    return (equalSpans(category, StrSpan("implication", 11))
            || equalSpans(category, StrSpan("existence", 9))
            || equalSpans(category, StrSpan("and", 3))
            || equalSpans(category, StrSpan("or", 2)))
        && name.len > category.len
        && equalSpans(StrSpan(name.ptr, category.len), category)
        && std::all_of(name.ptr + category.len, name.ptr + name.len,
                       [](unsigned char c) { return c >= '0' && c <= '9'; });
}

/// @brief Rewrite retired operator applications without changing surrounding text.
///
/// @details Only complete MPL atom applications match; scope delimiters and
/// binders pass through byte for byte. Substitutions compose transitively.
/// @param text Expression or proof field containing MPL applications.
/// @param replacements Retired-name substitution graph.
/// @return Text containing only retained applications.
/// @invariant Invalid permutations, malformed applications and cycles assert.
std::string rewriteCompilationReferences(const std::string& text,
                                        const CompilationReplacements& replacements) {
    if (replacements.empty()) return text;
    std::string result;
    result.reserve(text.size());
    for (std::size_t position = 0; position < text.size();) {
        if (text[position] != '(') {
            result += text[position++];
            continue;
        }
        std::size_t bracket = position + 1;
        while (bracket < text.size() && text[bracket] != '['
               && text[bracket] != '(' && text[bracket] != ')') ++bracket;
        const std::string name = text.substr(position + 1, bracket - position - 1);
        auto replacement = replacements.find(name);
        if (bracket == text.size() || text[bracket] != '[' || replacement == replacements.end()) {
            result += text[position++];
            continue;
        }
        const std::size_t close = text.find(']', bracket + 1);
        assert(close != std::string::npos && close + 1 < text.size()
               && text[close + 1] == ')' && "retired name must be a complete MPL atom");
        std::vector<std::string> arguments;
        if (close != bracket + 1) {
            assert(text[close - 1] != ',' && "compiled argument must not be empty");
            for (std::size_t start = bracket + 1; start < close;) {
                const std::size_t comma = text.find(',', start);
                const std::size_t end = std::min(comma, close);
                assert(end > start && "compiled argument must not be empty");
                arguments.push_back(text.substr(start, end - start));
                start = end + 1;
            }
        }
        std::string target = name;
        std::set<std::string> seen;
        while (replacement != replacements.end()) {
            assert(seen.insert(target).second && "compiled replacement cycle");
            const auto& instruction = replacement->second;
            assert(instruction.arguments.size() == arguments.size()
                   && "compiled replacement arity mismatch");
            std::vector<std::string> ordered;
            std::set<int> used;
            for (int index : instruction.arguments) {
                assert(index > 0 && static_cast<std::size_t>(index) <= arguments.size());
                assert(used.insert(index).second && "compiled replacement must be a permutation");
                ordered.push_back(arguments[static_cast<std::size_t>(index) - 1]);
            }
            arguments = std::move(ordered);
            target = instruction.target;
            replacement = replacements.find(target);
        }
        result += '(' + target + '[';
        for (std::size_t index = 0; index < arguments.size(); ++index) {
            if (index != 0) result += ',';
            result += arguments[index];
        }
        result += "])";
        position = close + 2;
    }
    return result;
}

/// @brief Install a reuse entry with public-name precedence.
///
/// @details Existing named winners are never replaced by another public name.
/// Generated-only registrations retain their established overwrite behavior.
/// @param key Normalized element vector and category.
/// @param value Signature mapping, core name and stored definition.
/// @param named Whether the registration declares a public name.
/// @return None.
void ExpressionAnalyzer::registerCompilationReuse(const CompilationReuseKey& key,
                                                  const CompilationReuseValue& value,
                                                  bool named) {
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

/// @brief Enumerate normalized reuse identities for a stored definition.
///
/// @details Permutes premises using the existing compiler machinery and keeps
/// an implication head last. Identity slots map the normalized key back to the
/// stored public signature, including nontrivial argument permutations.
/// @param name Existing compiled operator name.
/// @return Ordered normalized keys and signature-correct registration values.
std::vector<std::pair<ExpressionAnalyzer::CompilationReuseKey, ExpressionAnalyzer::CompilationReuseValue>>
ExpressionAnalyzer::compilationReuseEntries(const std::string& name) {
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
    auto combinations = giveAllSortedCombinations(tail);
    combinations.insert(combinations.begin(), tail);
    std::map<std::string, std::string> signature;
    for (const auto& argument : ce::getArgs(entity.signature)) signature[argument] = argument;
    std::vector<std::pair<CompilationReuseKey, CompilationReuseValue>> result;
    for (auto& combination : combinations) {
        combination.insert(combination.end(), head.begin(), head.end());
        const auto normalized = makeNormalizedEncodedKey(combination, 2);
        const auto elements = splitCompilationKey(combination, normalized.first.data);
        const auto mapping = createReverseUnchangeableMap(signature, normalized.second);
        assert(mapping.size() == signature.size() && "every public signature slot must occur in its body");
        result.emplace_back(CompilationReuseKey{elements, entity.category},
                            CompilationReuseValue{mapping, name, entity.elements});
    }
    return result;
}

/// @brief Restore a loaded public predicate's reuse identities.
///
/// @details The loaded entity remains intact; only derived reuse indexes change.
/// @param name Loaded public operator name.
/// @return None.
void ExpressionAnalyzer::registerLoadedNamedCompilation(const std::string& name) {
    for (const auto& entry : compilationReuseEntries(name))
        registerCompilationReuse(entry.first, entry.second, true);
}

/// @brief Record a generated operator's argument-correct replacement.
///
/// @details Maps equal normalized slots through the old and retained signatures.
/// A previously recorded replacement keeps its first public target.
/// @param numbered Existing generated registration.
/// @param named Retained public registration.
/// @return None.
/// @invariant New promotions happen before proof-state creation.
void ExpressionAnalyzer::retireNumberedCompilation(const CompilationReuseValue& numbered,
                                                   const CompilationReuseValue& named) {
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

/// @brief Complete compiled-name retirement before proof consumers start.
///
/// @details Public identities are compared against generated definitions without
/// changing unrelated generated reuse keys. Rewrites may expose a parent-body
/// equivalence, so removal and named-index rebuilding repeat to a fixed point.
/// @return None.
/// @invariant Retained definitions and cache values contain no retired references.
void ExpressionAnalyzer::finalizeCompilationReplacements() {
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
        }
        for (const auto& [name, replacement] : compilationReplacements) {
            (void)replacement;
            changed = compiledExpressions.erase(name) != 0 || changed;
            coreExpressionMap.erase(name);
            orsAwaitingImplications.erase(name);
            existencesAwaitingImplications.erase(name);
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
            std::vector<EncodedExpression> encoded;
            for (const auto& element : elements) encoded.emplace_back(element, "main");
            const auto normalized = makeNormalizedEncodedKey(encoded, 2);
            auto mapping = createReverseUnchangeableMap(std::get<0>(value), normalized.second);
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
            rewritten[{splitCompilationKey(encoded, normalized.first.data), key.second}] =
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
    }
}

/// @brief Restore monotone counters and replacement metadata from a sidecar.
///
/// @details Absence denotes the legacy registry format. Present state must have
/// schema 1 and complete, nonnegative counters and replacement permutations.
/// @param registryPath Registry file whose sibling state is read.
/// @return None.
void ExpressionAnalyzer::loadCompilationState(const std::filesystem::path& registryPath) {
    const std::string filename = registryPath.filename().string();
    assert(filename.rfind("GL_binary_", 0) == 0);
    const auto path = registryPath.parent_path() / ("GL_compilation_state_" + filename.substr(10));
    if (!std::filesystem::exists(path)) return;
    std::ifstream input(path);
    assert(input.is_open());
    nlohmann::json state;
    input >> state;
    assert(state.is_object() && state.size() == 3);
    assert(state.at("schema").is_number_integer() && state.at("schema") == 1);
    assert(state.at("next_counters").is_object() && state.at("next_counters").size() == 4);
    assert(state.at("replacements").is_object());
    int* counters[] = {&implCounter, &existenceCounter, &andCounter, &orCounter};
    const char* categories[] = {"implication", "existence", "and", "or"};
    for (int index = 0; index < 4; ++index) {
        const auto& number = state.at("next_counters").at(categories[index]);
        assert(number.is_number_integer() && number.get<int>() >= 0);
        *counters[index] = std::max(*counters[index], number.get<int>());
    }
    for (const auto& item : state.at("replacements").items()) {
        const auto& record = item.value();
        assert(record.is_object() && record.size() == 2);
        assert(record.at("target").is_string() && !record.at("target").get<std::string>().empty());
        assert(record.at("arguments").is_array());
        assert(std::any_of(std::begin(categories), std::end(categories),
            [&](const char* category) { return isNumberedCompilationName(item.key(), category); }));
        for (const auto& position : record.at("arguments"))
            assert(position.is_number_integer() && position.get<int>() > 0);
        CompilationReplacement replacement{record.at("target").get<std::string>(),
                                           record.at("arguments").get<std::vector<int>>()};
        const auto inserted = compilationReplacements.emplace(item.key(), replacement);
        assert(inserted.second || (inserted.first->second.target == replacement.target
                                   && inserted.first->second.arguments == replacement.arguments));
    }
}

/// @brief Save migration metadata and monotone next-number counters.
///
/// @details Uses a schema-1 sidecar; the compiled-definition JSON stays unchanged.
/// @param outDir Registry output directory.
/// @return None.
void ExpressionAnalyzer::exportCompilationState(const std::filesystem::path& outDir) {
    nlohmann::json state = {{"schema", 1},
        {"next_counters", {{"implication", implCounter}, {"existence", existenceCounter},
                           {"and", andCounter}, {"or", orCounter}}},
        {"replacements", nlohmann::json::object()}};
    for (const auto& [name, replacement] : compilationReplacements) {
        state["replacements"][name] = {{"target", replacement.target}, {"arguments", replacement.arguments}};
    }
    std::ofstream output(outDir / ("GL_compilation_state_" + anchorID_ + ".json"));
    assert(output.is_open());
    output << state.dump(2) << '\n';
    output.flush();
    assert(output.good());
}

} // namespace gl
