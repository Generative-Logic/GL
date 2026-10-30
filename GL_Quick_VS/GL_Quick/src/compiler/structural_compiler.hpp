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
class CompilationReuse;
class DefinitionExpansion;
class OrRuleCompiler;
class ExistenceRuleCompiler;
class SupportingRuleCompiler;
class IntegrationRuleCompiler;

/// @brief Compile source structure and prepare definition metadata.
/// @details Composes through definition interfaces without retaining a prover.
class StructuralCompiler {
public:
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param compilationReuse Borrowed compilationReuse component.
/// @param definitionExpansion Borrowed definitionExpansion component.
/// @param orRuleCompiler Borrowed orRuleCompiler component.
/// @param existenceRuleCompiler Borrowed existenceRuleCompiler component.
/// @param supportingRuleCompiler Borrowed supportingRuleCompiler component.
/// @param integrationRuleCompiler Borrowed integrationRuleCompiler component (the integration hooks).
/// @return Constructed compilation component.
StructuralCompiler(CompiledDefinitionRegistry& definitions, CompilationReuse& compilationReuse, DefinitionExpansion& definitionExpansion, OrRuleCompiler& orRuleCompiler, ExistenceRuleCompiler& existenceRuleCompiler, SupportingRuleCompiler& supportingRuleCompiler, IntegrationRuleCompiler& integrationRuleCompiler);
/// @brief Compile one source expression to its retained compact definition.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param inputExpression Source expression to compile.
/// @return The result of the documented operation.
std::string compileExpression(const std::string& inputExpression);
/// @brief Compile configured definitions and derive their type metadata.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @return None.
void compileCoreExpressionMap();
/// @brief Compile a mail-bound implication into its compact named
        /// form `(implication<N>[args])`.
        ///
        /// @details
        /// Wrapper invoked at every site where a proved or internal
        /// implication is broadcast on the mail "implications" channel.
        /// It normalises the implication's free arguments and delegates to
        /// the existing `compileExpression` implication branch
        /// (the `(>` branch — the only branch reached for a top-level
        /// non-negated implication), returning the compact instance.
        ///
        /// Free arguments arrive in `u_`-prefixed template form for
        /// internal implications; a fully-bound proved theorem carries no
        /// `u_` at all (every variable is bound — see
        /// [D-75](../../docs/agentic_swdd/40_decisions.md)). `stripUPrefixAST` removes a
        /// leading `u_` from every argument token at every nesting level,
        /// yielding a uniform bare-argument `(>[]()())` string before
        /// compilation. A fully-bound theorem then has zero free arguments
        /// and compiles to `(implication<N>[])`; an implication with
        /// genuinely-free arguments compiles to `(implication<N>[args])`.
        ///
        /// Preparatory for ASIC 0.1: the returned compact form is
        /// additionally deposited as a mail expression; it has no effect on
        /// which theorems prove.
        ///
        /// No defensive fallback (CLAUDE.md Rule 19): an implication that
        /// reached the mail channel and fails to compile to a simple
        /// `(implication<N>[...])` is a bug and asserts at its origin — it
        /// never returns an empty / sentinel string the caller would have
        /// to test-and-skip.
        ///
        /// @param implication Canonical MPL implication string — the
        ///        original expanded `(>[]()())` as it enters the mail
        ///        channel (the 5th element of a `mailOut.implications`
        ///        tuple).
        /// @return The compact `(implication<N>[args])` form;
        ///         `(implication<N>[])` for a fully-bound theorem.
        /// @invariant The result is non-empty, satisfies
        ///            `ce::expressionIsSimple`, and its core name begins
        ///            with `"implication"`.
        /// @see stripUPrefixAST, compileExpression,
        ///      excludeRepetitions
std::string compileImplicationToCompact(const std::string& implication);
/// @brief Perform precompileStructuralOperators using the registered definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param theorem Input to this compilation operation.
/// @return None.
void precompileStructuralOperators(std::string& theorem);
/// @brief Compile, once, the extensionality entity of a non-digit
        ///        definition set.
        ///
        /// @details
        /// Finds the membership operator (`in`, `in2`, `in3`) whose last slot
        /// carries @p label — asserts when none does (no extensionality for
        /// that type). Compiles its subset implication
        /// `(>[x..](m[x..,A])(m[x..,B]))` through the ordinary compact compiler
        /// (`compileImplicationToCompact`; an existing compact with the same
        /// body is reused, e.g. `implication37` for sets in FTA), with
        /// non-numeric names so no bound variable meets a free one. The compact
        /// instance's argument order fixes which formal is `A`. Runs only on
        /// the single-threaded registration seams, because compiling mints
        /// registry entries.
        ///
        /// @param label A definition set other than `(1)`.
        /// @invariant At most one entity per label.
        /// @see typedEqualityEntity, ensureTypedEqualityEntityForTheorem.
/// @return None.
void ensureTypedEqualityEntity(StrSpan label);

/// @brief Verify that compiled implication heads resolve to registered definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @return None.
void checkCompiledCoreExpressionMap();

private:
/// @brief Register configuration metadata for generated logical definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @return None.
void extendCoreExpressionMap();
/// @brief Resolve the definition set of a formal argument through compiled elements.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param var Formal argument to resolve.
/// @param le Compiled logical entity to inspect.
/// @return The result of the documented operation.
std::string findDefinitionSetRecursive(const std::string& var, const LogicalEntity& le);
CompiledDefinitionRegistry& definitions;
CompilationReuse& compilationReuse;
DefinitionExpansion& definitionExpansion;
OrRuleCompiler& orRuleCompiler;
ExistenceRuleCompiler& existenceRuleCompiler;
SupportingRuleCompiler& supportingRuleCompiler;
IntegrationRuleCompiler& integrationRuleCompiler;
};
} // namespace gl
