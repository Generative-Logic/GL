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
#include "../inference/hash_rule_installer.hpp"
#include <utility>
namespace gl::tests {
struct HashRuleInstallerFixture {
template<class... Args> static decltype(auto) addWholeKeyOwner(Args&&... args) { return HashRuleInstaller::addWholeKeyOwner(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) ensureEmptySubkeyRecord(Args&&... args) { return HashRuleInstaller::ensureEmptySubkeyRecord(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) extractRemainingArgs(Owner& target, Args&&... args) { return target.extractRemainingArgs(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) mergeSubkeySignatures(Args&&... args) { return HashRuleInstaller::mergeSubkeySignatures(std::forward<Args>(args)...); }
};
}
