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
#include "../inference/hash_rule_index.hpp"
#include <utility>
namespace gl::tests {
struct HashRuleIndexFixture {
template <typename... Args> static decltype(auto) flushLmvSection(Args&&... args) { return HashRuleIndex::flushLmvSection(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) flushOwnerSection(Args&&... args) { return HashRuleIndex::flushOwnerSection(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) flushRemArgsSection(Args&&... args) { return HashRuleIndex::flushRemArgsSection(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) flushSubkeySection(Args&&... args) { return HashRuleIndex::flushSubkeySection(std::forward<Args>(args)...); }
};
}
