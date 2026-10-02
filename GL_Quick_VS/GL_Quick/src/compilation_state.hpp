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

#include <map>
#include <string>
#include <vector>

namespace gl {

struct StrSpan;

/// @brief A removed compiled operator's retained name and argument permutation.
/// @details Position i of arguments selects the one-based old argument to put
/// in position i of the retained predicate. This is migration metadata only.
struct CompilationReplacement {
    std::string target;
    std::vector<int> arguments;
};

using CompilationReplacements = std::map<std::string, CompilationReplacement>;

/// @brief Distinguish generated operator names from public predicate names.
///
/// @details A generated name consists of its logical category followed by a
/// nonempty decimal suffix. A public name is the other defined result.
/// @param name Operator name.
/// @param category Logical category.
/// @return Whether the name has the generated spelling.
bool isNumberedCompilationName(const std::string& name, const std::string& category);

/// @brief Classify a compiled operator name without allocating string storage.
///
/// @details Reads the supplied spans synchronously. Only implication,
/// existence, conjunction and disjunction category names followed by a decimal suffix
/// are generated names; every other spelling is a public name.
/// @param name Operator name, without expression punctuation.
/// @param category Logical category.
/// @return Whether the name has the generated spelling.
/// @see isNumberedCompilationName(const std::string&, const std::string&).
bool isNumberedCompilationName(StrSpan name, StrSpan category);

/// @brief Rewrite retired operator applications without changing surrounding text.
///
/// @details Recognizes complete parenthesized MPL applications, including those
/// embedded in scope payloads. Applies one-based permutations transitively while
/// preserving all other bytes, variable names, binders, negations and delimiters.
/// @param text Expression or proof field containing MPL applications.
/// @param replacements Retired-name substitution graph.
/// @return Text containing only retained applications.
/// @invariant Cycles, malformed applications and invalid permutations assert.
std::string rewriteCompilationReferences(const std::string& text,
                                        const CompilationReplacements& replacements);

} // namespace gl
