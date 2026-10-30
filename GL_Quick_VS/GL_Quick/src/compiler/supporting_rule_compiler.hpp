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

namespace gl {
class CompiledDefinitionRegistry;
class StructuralCompiler;

/// @brief Compile contraposition, conjunction refutation and supporting rule families.
/// @details Composes through definition interfaces without retaining a prover.
class SupportingRuleCompiler {
public:
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param structuralCompiler Borrowed structuralCompiler component.
/// @return Constructed compilation component.
SupportingRuleCompiler(CompiledDefinitionRegistry& definitions, StructuralCompiler& structuralCompiler);
/// @brief Prepare every binder-covering contraposition in compiled families.
        /// @details Uses only single-premise disintegration products. Conjunction
        /// families retain their source order and every covering subset; generated
        /// auxiliaries never become new defining elements or generation families.
        /// @return Void.
        /// @invariant Compilation and canonical reference completion precede workers.
void compileContrapositionFamilies();
/// @brief Compile fixed-parameter refutations of conjunction members.
        /// @details Prepare direct and nested member templates independently of
        /// runtime facts. Mark eligible prepared reversals for scoped discovery.
        /// Defining elements and integration instructions remain unchanged.
        /// @return Void.
        /// @invariant Called only at single-threaded compilation seams.
        /// @see compileContrapositionFamilies.
void compileConjunctionRefutations();
/// @brief Compile guarded helpers for negated immediate conjunction premises.
        /// @details Preserve fixed parameters and complete premise binding; generated
        /// helpers do not recursively seed either auxiliary preparation family.
        /// Unary owners install direct compositions with proof-only intermediates.
        /// @return Void.
        /// @invariant Called only at single-threaded compilation seams.
void compileSupportingImplications();
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
void compileDisintegrationPackages();
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
void computeDisintegrationPackage(const LogicalEntity& entity, std::vector<std::string>& binders,
    std::vector<std::string>& placeholders, std::vector<DisintegrationShot>& shots) const;
/// @brief Recognize the unary negative-typing rule excluded by D-382.
        /// @details The head denies membership of the sole bound variable in a
        /// fixed set. Reversing this negated-existence body-rule shape would
        /// restore the deliberately suppressed typing-to-negative-body direction.
        /// @param entity Compiled candidate implication.
        /// @return True for precisely the excluded negative-typing rule shape.
static bool isContrapositionTypingExclusion(const LogicalEntity& entity) {
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

public:
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
static bool isContrapositionUncoveredDefinition(const LogicalEntity& entity, bool publicDefinition);

/// @brief Visit every compiled definition in map order, saying whether it was
///        added since a pass recorded the entries present.
///
/// @details
/// The three preparation passes record the address of every entry's entity
/// while they collect their owners, then mark the entries they added. The
/// registry is a node-based map the passes only extend — an entry keeps its
/// address while new ones are inserted around it, and nothing is erased — so
/// one walk in map order alongside the recorded addresses tells them apart:
/// an entry whose address is the next recorded one was present, any other
/// entry was added. This is the answer a set of the names present at the
/// recording gives, without copying a name. Every recorded address must be
/// met (asserted): an erased entry would break the correspondence.
///
/// @tparam Fn Callable `void(const std::string& name, LogicalEntity& entity, bool added)`.
/// @param map     The compiled-definition map.
/// @param present The entity addresses recorded in map order on the same map.
/// @param fn      The visitor, called once per entry in map order.
/// @return Nothing.
/// @see compileContrapositionFamilies, compileConjunctionRefutations,
///      compileSupportingImplications
template <typename Fn>
static void forEachEntryWithAdded(CompiledExpressionMap& map,
        const std::vector<const LogicalEntity*>& present, Fn fn) {
    std::size_t next = 0;
    for (auto& [name, entity] : map) {
        const bool recorded = next < present.size() && present[next] == &entity;
        if (recorded) ++next;
        fn(name, entity, !recorded);
    }
    assert(next == present.size()
        && "forEachEntryWithAdded: an entry present at the recording is gone");
}
private:
CompiledDefinitionRegistry& definitions;
StructuralCompiler& structuralCompiler;
};
} // namespace gl
