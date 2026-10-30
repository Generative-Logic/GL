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
#include "compilation_normalizer.hpp"
#include "../compilation_state.hpp"

namespace gl {
class CompiledDefinitionRegistry;

using CompilationReuseKey = std::pair<std::vector<std::string>, std::string>;
using CompilationReuseValue = std::tuple<std::map<std::string, std::string>, std::string, std::vector<std::string>>;
using CompilationReuseMap = std::map<CompilationReuseKey, CompilationReuseValue>;

/// @brief The value state of `CompilationReuse`: every reuse index, the
///        replacement metadata and the public-name phase flag.
/// @details Separated from the component's bound references so that the
/// complete state can be read and replaced as one value
/// (`CompilationReuse::state`, `CompilationReuse::restoreState`). A member
/// added here is carried by every state copy automatically.
struct CompilationReuseState {
    CompilationReuseMap repetitionExclusionMap;
    CompilationReuseMap namedRepetitionExclusionMap;
    std::map<std::vector<std::string>, std::pair<std::string, std::vector<int>>> compiledOrReuse;
    std::vector<std::string> namedCompilationOrder;
    std::set<std::string> namedCompilationNames;
    CompilationReplacements compilationReplacements;
    bool compilationNamesFrozen = false;
};

/// @brief Own normalized definition reuse and public-name replacement state.
/// @details Binds only the definition tables, permutation table and pending
/// definition-family sets needed by reuse; no prover object is retained.
class CompilationReuse : private CompilationReuseState {
public:
/// @brief Bind the definition stores used by reuse operations.
/// @details References remain valid for this compiler instance lifetime.
/// @param definitions Definition registry supplying the private stores.
/// @param permutations Prepared ordered premise permutations.
/// @return Constructed reuse component.
CompilationReuse(CompiledDefinitionRegistry& definitions,
    std::map<int, std::vector<std::vector<int>>>& permutations);

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
std::string excludeRepetitions(const std::vector<EncodedExpression>& encodedTail, const std::vector<EncodedExpression>& encodedHead, const std::string& newCoreExpressionName, const std::string& category, const bool force, const std::string defaultSignature);

/// @brief Find a compiled OR by its ordered defining elements.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param elems Ordered defining elements.
/// @param elemN Number of elements.
/// @param argumentPositions Optional destination for retained argument positions.
/// @param capacity Capacity of the optional destination.
/// @return Stable retained name, or null for the defined no-match case.
const std::string* compiledOrByElements(const StrSpan* elems, int32_t elemN, int* argumentPositions = nullptr, int capacity = 0) const;

/// @brief Split a normalized key using the source expression arities.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param expressions Source expressions carrying their arities.
/// @param normalized Normalized key tokens in source order.
/// @return Ordered normalized expressions.
static std::vector<std::string> splitCompilationKey(const std::vector<EncodedExpression>& expressions, const std::vector<std::string>& normalized);

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
static std::pair<std::vector<std::string>, std::map<std::string, std::string>> canonicalReuseKey(
    const std::vector<EncodedExpression>& tail, const std::vector<EncodedExpression>& head);

/// @brief Register a reuse identity with existing public-name precedence.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param key Normalized body and logical category.
/// @param value Argument mapping, retained name and defining elements.
/// @param named Whether this is a public definition.
/// @return None.
void registerCompilationReuse(const CompilationReuseKey& key, const CompilationReuseValue& value, bool named);

/// @brief Restore the reuse identities of one loaded public definition.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param name Loaded public definition.
/// @return None.
void registerLoadedNamedCompilation(const std::string& name);

/// @brief Rewrite and retire numbered definitions before proof consumers start.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @return None.
void finalizeCompilationReplacements();

    /// @brief Register the normalized identities of an already stored definition.
    /// @details Preserves the identity enumeration and public-name precedence order.
    /// @param name Stored definition to register.
    /// @param named Whether the definition carries a public name.
    /// @return None.
    void registerStoredCompilation(const std::string& name, bool named);

    /// @brief Read the normalized reuse identities.
    /// @details The view remains owned by this component and permits no mutation.
    /// @return Read-only reuse map.
    const CompilationReuseMap& entries() const { return repetitionExclusionMap; }
    /// @brief Read the public-name reuse identities.
    /// @details The first public registration retains precedence for each body.
    /// @return Read-only public reuse map.
    const CompilationReuseMap& publicEntries() const { return namedRepetitionExclusionMap; }
    /// @brief Read generated-name replacement metadata.
    /// @details Consumers use this view to migrate their expression references.
    /// @return Read-only replacement map.
    const CompilationReplacements& replacements() const { return compilationReplacements; }
    /// @brief Finish the phase that permits public-name promotion.
    /// @details Repeated calls retain the frozen state without changing records.
    /// @return None.
    void freezePublicNames() { compilationNamesFrozen = true; }
    /// @brief Report whether proof consumers have frozen public-name promotion.
    /// @details This query changes no compilation state.
    /// @return Whether public-name promotion is frozen.
    bool publicNamesFrozen() const { return compilationNamesFrozen; }
    /// @brief Retain a public OR's argument mapping for an ordered body.
    /// @details Replaces the same body's cached mapping in its existing order.
    /// @param elements Ordered normalized OR elements.
    /// @param name Retained public predicate.
    /// @param positions Argument positions in the retained predicate.
    /// @return None.
    void rememberOrReuse(const std::vector<std::string>& elements,
        const std::string& name, const std::vector<int>& positions);
    /// @brief Restore one generated-name replacement from compilation input.
    /// @details A repeated replacement must agree exactly with its stored row.
    /// @param name Generated name to replace.
    /// @param replacement Retained name and argument permutation.
    /// @return None.
    void restoreReplacement(const std::string& name, const CompilationReplacement& replacement);
    /// @brief Clear reuse indexes and replacement metadata for a fresh compiler input.
    /// @details Definition stores and the public-name phase state are unchanged.
    /// @return None.
    void clear();
    /// @brief Read the complete value state of this reuse component.
    ///
    /// @details
    /// The view covers every reuse index, the replacement metadata and the
    /// public-name phase flag; the bound definition stores are not part of
    /// it. A copy of the view is what `restoreState` accepts, so a compiled
    /// configuration can be recorded once and installed into another
    /// component bound to an equal registry.
    ///
    /// @return Read-only view owned by this component.
    /// @see restoreState
    const CompilationReuseState& state() const { return *this; }
    /// @brief Replace the complete value state with a recorded one.
    ///
    /// @details
    /// Every reuse index, the replacement metadata and the public-name phase
    /// flag become copies of @p recorded; the bound definition stores are not
    /// touched. The caller installs the matching registry, so the restored
    /// indexes name the entities they were recorded with.
    ///
    /// @param recorded State read through `state` from a component whose
    ///        registry equals this component's registry.
    /// @return None.
    /// @see state
    void restoreState(const CompilationReuseState& recorded);

private:

/// @brief The reuse identity of a compiled definition.
///
/// @details One entry: the canonical key of the definition's elements
/// (`canonicalReuseKey`, an implication's last element as head) with the map
/// from its canonical slots to the definition's own signature slots.
/// @param name Registered definition to inspect.
/// @return The definition's single reuse identity.
std::vector<std::pair<CompilationReuseKey, CompilationReuseValue>> compilationReuseEntries(const std::string& name);

/// @brief Record the argument-correct replacement of a numbered definition.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param numbered Numbered definition being replaced.
/// @param named Retained public definition.
/// @return None.
void retireNumberedCompilation(const CompilationReuseValue& numbered, const CompilationReuseValue& named);

/// @brief Reconstruct ordered expression strings from a normalized key.
///
/// @details Preserves category separation, registration order and argument mappings.
/// @param normalizedKey Tokens with operator, polarity and normalized arguments.
/// @return Ordered expression strings.
std::vector<std::string> splitNormalizedKey(const std::vector<std::string> normalizedKey);

    CompiledExpressionMap& compiledExpressions;
    ce::CoreExpressionMap& coreExpressionMap;
    std::map<int, std::vector<std::vector<int>>>& allPermutationsAna;
};

} // namespace gl
