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
#include "../logical_block/demand_built_rules.hpp"
#include <mutex>
namespace gl::tests {
struct DemandBuiltRuleRegistryFixture {
/// @brief Forget every registered rule: the unit tests of one process share
///        the process-wide registry.
/// @return Nothing.
static void clear() {
    DemandBuiltRuleRegistry& registry = DemandBuiltRuleRegistry::instance();
    std::lock_guard<std::mutex> lock(registry.mutex_);
    registry.count_.store(0, std::memory_order_release);
    registry.anyOr_.store(false, std::memory_order_release);
}
};
}
