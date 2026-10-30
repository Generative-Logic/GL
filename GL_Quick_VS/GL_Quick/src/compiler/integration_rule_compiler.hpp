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
#include "compiled_definition_registry.hpp"

#include <string>
#include <vector>

namespace gl {
class StructuralCompiler;
class IntegrationInstructionBuilder;

/// @brief Compile the integration set of a logical entity — every rule
///        `prepareIntegration` installs for an instance of the entity, as
///        compact implications over the entity's own tokens.
/// @details Composes through definition interfaces without retaining a prover.
class IntegrationRuleCompiler {
public:
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param structuralCompiler Borrowed structuralCompiler component (compacts are compiled through it).
/// @param integrationBuilder Borrowed integration instruction builder (builds the definition table).
/// @return Constructed compilation component.
IntegrationRuleCompiler(CompiledDefinitionRegistry& definitions, StructuralCompiler& structuralCompiler,
    IntegrationInstructionBuilder& integrationBuilder);

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
void compileIntegrationImplications(const std::string& name);

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
/// definition tree are bound inside each compact.
///
/// @param entity         The entity; category `and` or `existence` (asserted).
/// @param registryEntity True for an entity of the registry (the definition
///                       table resolves it by name); false for an entity
///                       outside it (the extensionality entity of a typed `=`,
///                       I-267), which the table instantiates directly.
/// @return The integration set.
/// @see compileIntegrationImplications, StructuralCompiler::ensureTypedEqualityEntity.
std::vector<IntegrationImplication> compileIntegrationFamily(const LogicalEntity& entity, bool registryEntity);

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
std::string expandIntegrationCompact(const std::string& compact) const;

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
static bool sameUpToBoundNames(const std::string& a, const std::string& b);

private:
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
std::string compactOf(const std::vector<std::string>& premises, const std::string& head,
    int32_t tokenCount);

CompiledDefinitionRegistry& definitions;
StructuralCompiler& structuralCompiler;
IntegrationInstructionBuilder& integrationBuilder;
};
} // namespace gl
