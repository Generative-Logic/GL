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

/// @brief Load and save the existing compiled-definition artifact formats.
/// @details Composes through definition interfaces without retaining a prover.
class CompilationPersistence {
public:
/// @brief Bind the cooperating compilation components.
/// @details Construction stores references only; all bound objects outlive this component.
/// @param definitions Borrowed definitions component.
/// @param compilationReuse Borrowed compilationReuse component.
/// @param anchorID_ Borrowed anchorID_ component.
/// @return Constructed compilation component.
CompilationPersistence(CompiledDefinitionRegistry& definitions, CompilationReuse& compilationReuse, const std::string& anchorID_);
/// @brief Perform loadGlBinary using the registered definitions.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param jsonPath Input to this compilation operation.
/// @return None.
void loadGlBinary(const std::filesystem::path& jsonPath);
/// @brief Serialize compiled definitions and their compilation-state sidecar.
///
/// @details Preserves expression order, binding and compiler name allocation.
/// @param outDir Output directory for the batch registry.
/// @return None.
void exportCompiledExpressionsJSON(const std::filesystem::path& outDir);
/// @brief Restore monotone compiler counters and replacement metadata.
        ///
        /// @details A missing sidecar denotes the original registry format;
        /// registry-derived counters remain valid for that format. Present
        /// sidecars are strictly validated before their state is applied.
        /// @param registryPath Compiled registry whose sibling state is loaded.
        /// @return None.
void loadCompilationState(const std::filesystem::path& registryPath);
/// @brief Save the compiled registry's migration state and next counters.
        ///
        /// @details Emits schema 1 beside the existing unchanged registry JSON.
        /// Counters remain above all issued names, including removed names.
        /// @param outDir Registry output directory.
        /// @return None.
void exportCompilationState(const std::filesystem::path& outDir);

private:
CompiledDefinitionRegistry& definitions;
CompilationReuse& compilationReuse;
const std::string& anchorID_;
};
} // namespace gl
