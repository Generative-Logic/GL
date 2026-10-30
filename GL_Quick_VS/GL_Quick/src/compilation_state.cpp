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





} // namespace gl
