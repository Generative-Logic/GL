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

/// @brief Own compiled definitions, configuration metadata and monotone name counters.
/// @details Composes through definition interfaces without retaining a prover.
class CompiledDefinitionRegistry {
private:
struct TypedEqualityEntity {
            std::string label;
            LogicalEntity entity;
        };
public:
/// @brief Zero-allocation read-fence accessor: the read side of
        ///        `coreExpressionMap`. That map still lives on the heap — an OPEN
        ///        violation to statify, NOT a sanctioned heap island / carve-out;
        ///        this accessor is only the zero-allocation read fence over it.
        ///
        /// @details
        /// Routes a phase-tree point read through the sole reader accessor
        /// instead of a materialized-`std::string` `coreExpressionMap.find`.
        /// The span converts to a `std::string_view` and probes the
        /// transparent (`std::less<>`) map heterogeneously — NO `std::string`
        /// is ever built to key the lookup.
        ///
        /// Read-out ONLY: a transparent `find` and a pointer return. No
        /// transformation, no caching / memoization, no side effects, no
        /// logging — any future urge to add those is a Rule-8
        /// conversation, not an edit.
        ///
        /// `nullptr` is the DEFINED miss — the exact analogue of
        /// `find() == end()`. This is the sanctioned bivalent contract
        /// (Rule 19): at every converted site the miss branch is a defined
        /// result — either an existing `if (it == end())` branch or an
        /// existing `assert`; S6b preserves each site's own verdict handling.
        ///
        /// The empty-span guard maps `StrSpan()` (e.g. `extractExpressionSpan`
        /// on a bracketless input) to the same DEFINED miss `find("")`
        /// produced — `""` is never a key; it also sidesteps
        /// `std::string_view(nullptr, 0)` pedantry. Not defensive: both
        /// branches are defined results (Rule 19 bivalent-contract test).
        ///
        /// The returned pointer aliases the map node — stable for the
        /// program's life on the read side (node-based `std::map`;
        /// phase-parallel sweeps are read-only because writers run
        /// only on single-threaded seams, and no converted site holds the
        /// pointer across such a seam — each consumes it synchronously).
        ///
        /// The `std::less<>` byte-order contract (span probe returns exactly
        /// the entry a materialized-`std::string` probe returned) lives on
        /// `ce::CoreExpressionMap`.
        ///
        /// @param core Span over the core-expression name to probe.
        /// @return Pointer to the map entry, or `nullptr` (defined miss).
        /// @invariant `I-137`.
        /// @see `compiledEntity` — the sibling accessor.
        /// @see `gl::extractExpressionSpan`, `ce::extractExpression`.
const ce::CoreExpressionConfig* coreConfig(const StrSpan& core) const {
            const std::string_view sv = (core.len > 0)
                ? std::string_view(core.ptr, static_cast<std::size_t>(core.len))
                : std::string_view();
            const auto it = coreExpressionMap.find(sv);
            return (it == coreExpressionMap.end()) ? nullptr : &it->second;
        }
/// @brief Zero-allocation read-fence accessor: the read side of
        ///        `compiledExpressions`. That map still lives on the heap — an OPEN
        ///        violation to statify, NOT a sanctioned heap island / carve-out;
        ///        this accessor is only the zero-allocation read fence over it.
        ///
        /// @details
        /// Routes a phase-tree point read through the sole reader accessor
        /// instead of a materialized-`std::string` `compiledExpressions.find`.
        /// The span converts to a `std::string_view` and probes the
        /// transparent (`std::less<>`) map heterogeneously — NO `std::string`
        /// is ever built to key the lookup.
        ///
        /// Read-out ONLY: a transparent `find` and a pointer return. No
        /// transformation, no caching / memoization, no side effects, no
        /// logging — any future urge to add those is a Rule-8
        /// conversation, not an edit.
        ///
        /// `nullptr` is the DEFINED miss — the exact analogue of
        /// `find() == end()`. This is the sanctioned bivalent contract
        /// (Rule 19): at every converted site the miss branch is a defined
        /// result — either an existing `if (it == end())` branch or an
        /// existing `assert`; S6b preserves each site's own verdict handling.
        ///
        /// The empty-span guard maps `StrSpan()` (e.g. `extractExpressionSpan`
        /// on a bracketless input) to the same DEFINED miss `find("")`
        /// produced — `""` is never a key; it also sidesteps
        /// `std::string_view(nullptr, 0)` pedantry. Not defensive: both
        /// branches are defined results (Rule 19 bivalent-contract test).
        ///
        /// The returned pointer aliases the map node — stable for the
        /// program's life on the read side (node-based `std::map`;
        /// phase-parallel sweeps are read-only because writers run
        /// only on single-threaded seams, and no converted site holds the
        /// pointer across such a seam — each consumes it synchronously).
        ///
        /// The `std::less<>` byte-order contract (span probe returns exactly
        /// the entry a materialized-`std::string` probe returned) lives on
        /// `gl::CompiledExpressionMap`.
        ///
        /// @param core Span over the core-expression name to probe.
        /// @return Pointer to the map entry, or `nullptr` (defined miss).
        /// @invariant `I-137`.
        /// @see `coreConfig` — the sibling accessor.
        /// @see `gl::extractExpressionSpan`, `ce::extractExpression`.
const LogicalEntity* compiledEntity(const StrSpan& core) const {
            const std::string_view sv = (core.len > 0)
                ? std::string_view(core.ptr, static_cast<std::size_t>(core.len))
                : std::string_view();
            const auto it = compiledExpressions.find(sv);
            return (it == compiledExpressions.end()) ? nullptr : &it->second;
        }
/// @brief Whether an expression is a positive instance of a configured
        ///        definition whose whole body is one implication.
        ///
        /// @details
        /// Such a definition (`primeseq`, `subset`) is compiled as category
        /// `implication`, and its public name replaces every numbered
        /// equivalent (D-349): it is itself the compact form of its rule
        /// (D-467). Wherever a numbered `(implication<N>[…])` counts as an
        /// implication — a rule carrier, an and element that never gates —
        /// this instance counts as one too. "Configured" is
        /// `expressionsFromConfig` membership; an atomic configured operator
        /// has no compiled entity. A negated instance `!(…)` is no
        /// implication. Zero allocation: the name is a span of the input and
        /// both probes are transparent lookups.
        ///
        /// @param expression Span over one flat expression.
        /// @return `true` exactly for `(name[…])` with `name` configured and
        ///         compiled as category `implication`.
        /// @see `compiledEntity`, `configuredNames`,
        ///      `StatementMemory::isConfiguredImplicationCarrier`.
bool isConfiguredImplication(const StrSpan& expression) const {
            if (expression.len < 3 || expression.ptr[0] != '(') return false;
            const StrSpan name = extractExpressionSpan(expression);
            if (expressionsFromConfig.find(std::string_view(name.ptr,
                    static_cast<std::size_t>(name.len))) == expressionsFromConfig.end())
                return false;
            const LogicalEntity* entity = compiledEntity(name);
            return entity != nullptr && entity->category == "implication";
        }
/// @brief Perform forEachCompiledOr using the registered definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param fn Read-only visitor; true stops the scan.
/// @return None.
template <typename Fn>
void forEachCompiledOr(Fn&& fn) const {
            for (const auto& [name, le] : compiledExpressions) {
                if (le.category != "or") continue;
                if (fn(name, le)) return;
            }
        }
/// @brief The extensionality entity of a definition set.
        ///
        /// @details A linear scan (at most one entry per membership operator).
        /// Absence is a defined answer: the set was never registered.
        ///
        /// @param label The definition set.
        /// @return The entity, or `nullptr` when none is registered.
        /// @see ensureTypedEqualityEntity.
const LogicalEntity* typedEqualityEntity(StrSpan label) const {
            for (const TypedEqualityEntity& t : typedEqualityEntities)
                if (equalSpans(StrSpan(t.label), label)) return &t.entity;
            return nullptr;
        }
/// @brief Look up a core expression's definition-set entry for a 1-based
        ///        argument slot — the heap-free replacement for the
        ///        `std::to_string(idx) + definitionSets.find(...)` idiom.
        ///
        /// @details
        /// `CoreExpressionConfig::definitionSets` is a
        /// `std::map<std::string, std::pair<std::string, bool>>` keyed by the
        /// 1-based slot index as a decimal STRING ("1", "2", …). This helper
        /// writes the decimal key into a stack `char[16]` (`writeDecimalDigits`,
        /// byte-identical to `std::to_string` for non-negative ints) and
        /// linear-scans the (small) map comparing keys via `equalSpans`, so no
        /// probe `std::string` is materialized. The map is never mutated during
        /// the burst, so the returned pointer into its node is stable
        /// (`checkNecessityForEquality`, `disintegrateExprHypothetically` read it
        /// synchronously). A miss returns `nullptr` — a DEFINED result mirroring
        /// `find() == end()`, never a defensive fallback (Rule 19).
        ///
        /// @param cfg             The core expression config (non-null by caller
        ///                        contract; asserted).
        /// @param argIndexOneBased The 1-based argument slot.
        /// @return Pointer to the `(pattern, mandatory)` value, or `nullptr` on a
        ///         defined miss.
        /// @see writeDecimalDigits, equalSpans,
        ///      CoreExpressionConfig::definitionSets.
static const std::pair<std::string, bool>* findDefinitionSet(
            const ce::CoreExpressionConfig* cfg, int argIndexOneBased) {
            assert(cfg != nullptr && "findDefinitionSet: null config");
            char dec[16];
            const int32_t n = writeDecimalDigits(dec, argIndexOneBased);
            const StrSpan key(dec, n);
            for (const auto& kv : cfg->definitionSets) {
                if (equalSpans(StrSpan(kv.first), key)) return &kv.second;
            }
            return nullptr;
        }

/// @brief Read coreExpressionMap without exposing mutation.
/// @details The view remains valid until the next single-threaded compilation seam.
/// @return Read-only compilation metadata.
const ce::CoreExpressionMap& configurations() const { return coreExpressionMap; }
/// @brief Read compiledExpressions without exposing mutation.
/// @details The view remains valid until the next single-threaded compilation seam.
/// @return Read-only compilation metadata.
const CompiledExpressionMap& entries() const { return compiledExpressions; }
/// @brief Read operators without exposing mutation.
/// @details The view remains valid until the next single-threaded compilation seam.
/// @return Read-only compilation metadata.
const std::set<std::string, std::less<>>& operatorNames() const { return operators; }
/// @brief Read expressionsFromConfig without exposing mutation.
/// @details The view remains valid until the next single-threaded compilation seam.
/// @return Read-only compilation metadata.
const std::set<std::string, std::less<>>& configuredNames() const { return expressionsFromConfig; }
/// @brief Read baseExpressionNames without exposing mutation.
/// @details The view remains valid until the next single-threaded compilation seam.
/// @return Read-only compilation metadata.
const std::set<std::string>& baseNames() const { return baseExpressionNames; }
/// @brief Read the next implCounter allocation position.
/// @details Reading does not allocate a name.
/// @return Next monotone counter value.
int nextImplication() const { return implCounter; }
/// @brief Read the next existenceCounter allocation position.
/// @details Reading does not allocate a name.
/// @return Next monotone counter value.
int nextExistence() const { return existenceCounter; }
/// @brief Read the next andCounter allocation position.
/// @details Reading does not allocate a name.
/// @return Next monotone counter value.
int nextConjunction() const { return andCounter; }
/// @brief Read the next orCounter allocation position.
/// @details Reading does not allocate a name.
/// @return Next monotone counter value.
int nextOr() const { return orCounter; }
/// @brief Read the next variableCounter allocation position.
/// @details Reading does not allocate a name.
/// @return Next monotone counter value.
int nextVariable() const { return variableCounter; }
/// @brief Install configured source definitions and classify operators.
/// @details This runs after loading prior compiled definitions and before source compilation.
/// @param configurations Source configuration map.
/// @return None.
void configure(ce::CoreExpressionMap configurations);

private:


ce::CoreExpressionMap coreExpressionMap;
CompiledExpressionMap compiledExpressions;
        // std::less<> (transparent) so span-fed gates can probe with a
        // std::string_view — the ordering is byte-identical to std::less<
        // std::string>, so nothing observable moves.
std::set<std::string, std::less<>> operators;
        // std::less<> (transparent) so the prepareIntegrationCore2 core-expression
        // probes can lookup with a std::string_view (0% heap). This is a genuine
        // snapshot of the coreExpressionMap keys taken BEFORE extendCoreExpressionMap
        // / the OR-core insert (which add keys later), so it is a strict SUBSET of
        // coreConfig -- NOT interchangeable with a coreConfig fence, which would
        // widen membership. The transparent ordering is byte-identical to
        // std::less<std::string>, so the set's contents are unchanged.



std::set<std::string, std::less<>> expressionsFromConfig;
std::set<std::string> baseExpressionNames;
std::vector<TypedEqualityEntity> typedEqualityEntities;
int implCounter = 0;
int existenceCounter = 0;
int andCounter = 0;
int orCounter = 0;
int variableCounter = 0;
friend class StructuralCompiler;
friend class OrRuleCompiler;
friend class ExistenceRuleCompiler;
friend class SupportingRuleCompiler;
friend class IntegrationRuleCompiler;
friend class CompilationPersistence;
friend class CompilationReuse;
friend struct CompiledDefinitionFixture;
};
} // namespace gl
