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

namespace gl {

/// @brief Compiler argument normalization and canonical premise order.
/// @details Stateless algorithms retain their existing ordering and key spelling.
class CompilationNormalizer {
public:

/// @brief Build the retained definition argument list from its slot mapping.
///
/// @details Orders arguments by their numeric unchangeable slots and removes exactly one outer unchangeable prefix.
/// @param mp Original-to-normalized argument mapping.
/// @return Ordered bracketed argument list.
static std::string createArgList(const std::map<std::string, std::string>& mp);

/// @brief Compose argument renamings through normalized unchangeable slots.
///
/// @details Preserves unchanged ordinary arguments and requires every intermediate unchangeable slot to have a destination.
/// @param a First argument mapping.
/// @param b Second argument mapping.
/// @return Composed argument mapping.
static std::map<std::string, std::string> combineUnchangeablePaths(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& b);

/// @brief Relate normalized unchangeable slots to retained arguments.
///
/// @details Visits the second mapping in its existing key order and resolves each shared original through the first mapping.
/// @param a Retained mapping from original arguments.
/// @param b New mapping from the same original arguments.
/// @return New normalized slots mapped to retained arguments.
static std::map<std::string, std::string> createReverseUnchangeableMap(const std::map<std::string, std::string>& a, const std::map<std::string, std::string>& b);

/// @brief Order a body's premises canonically: by operator name, then the
///        smallest normalized form inside each same-name group.
///
/// @details
/// The repetition map keys every definition by this order, so two bodies that
/// differ only in premise order and variable names share one key. Premises are
/// grouped by operator name in string order. Inside the groups the order whose
/// normalized token sequence is lexicographically smallest wins: each premise
/// contributes its polarity and, per argument, its kind (a bound variable
/// before a `u_` one) and its number. Bound and `u_` variables are numbered
/// separately by first appearance, as `makeNormalizedEncodedKey` mode 2 numbers
/// them, and numbers compare numerically. The head follows the premises and
/// takes part in the comparison, so it decides ties the premises leave.
///
/// The minimum is found depth first, one premise slot at a time. At each slot
/// only the candidates with the smallest next token can lead to the minimum,
/// and a branch whose sequence exceeds the best complete one found so far is
/// dropped. Two tied candidates whose new variables differ only in variables
/// that occur in no other element are interchangeable: swapping them, together
/// with those private variables, maps the body onto itself and fixes
/// everything already placed. So among tied candidates with the same new
/// shared variables (each with its number) only the first is expanded. The
/// search visits at most every ordering of each group; it reaches that bound
/// only for premises that stay interchangeable to the end through shared
/// variables.
///
/// Among orderings with the minimal sequence, the first in input index order is
/// returned, so a list already in canonical order comes back unchanged.
///
/// @param tail Premises (every element of a body without a head), any order.
/// @param head Elements that follow the premises unpermuted: the conclusion of
///        an implication, empty otherwise.
/// @return The premises in canonical order; empty for an empty `tail`.
/// @invariant Premises with the same operator name have the same arity
///            (asserted).
/// @see makeNormalizedEncodedKey, CompilationReuse::canonicalReuseKey
static std::vector<EncodedExpression> canonicalCombination(const std::vector<EncodedExpression>& tail,
                                                           const std::vector<EncodedExpression>& head);

/// @brief Normalize expression arguments in first-occurrence order.
///
/// @details Preserves operator names, polarity and the existing unchangeable-argument modes. The returned mapping records the original-to-normalized argument correspondence.
/// @param lst Expressions in their existing order.
/// @param uStatus Existing unchangeable mode: normalize together, retain literal names, or normalize separately.
/// @return Normalized key and argument mapping.
static std::pair<NormalizedKey, std::map<std::string, std::string>> makeNormalizedEncodedKey(const std::vector<EncodedExpression>& lst, int uStatus = 1);
};

} // namespace gl
