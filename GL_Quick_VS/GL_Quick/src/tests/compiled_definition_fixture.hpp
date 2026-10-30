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
#include "../compiler/compiled_definition_registry.hpp"
namespace gl {
struct CompiledDefinitionFixture {
static auto& coreExpressionMap(CompiledDefinitionRegistry& registry) { return registry.coreExpressionMap; }
static auto& compiledExpressions(CompiledDefinitionRegistry& registry) { return registry.compiledExpressions; }
static auto& operators(CompiledDefinitionRegistry& registry) { return registry.operators; }
static auto& expressionsFromConfig(CompiledDefinitionRegistry& registry) { return registry.expressionsFromConfig; }
static auto& typedEqualityEntities(CompiledDefinitionRegistry& registry) { return registry.typedEqualityEntities; }
static auto& baseExpressionNames(CompiledDefinitionRegistry& registry) { return registry.baseExpressionNames; }
static auto& implCounter(CompiledDefinitionRegistry& registry) { return registry.implCounter; }
static auto& existenceCounter(CompiledDefinitionRegistry& registry) { return registry.existenceCounter; }
static auto& andCounter(CompiledDefinitionRegistry& registry) { return registry.andCounter; }
static auto& orCounter(CompiledDefinitionRegistry& registry) { return registry.orCounter; }
static auto& variableCounter(CompiledDefinitionRegistry& registry) { return registry.variableCounter; }
};
}
