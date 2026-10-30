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
#include "../logical_block/implication_subproofs.hpp"
#include <utility>
namespace gl::tests {
struct ImplicationSubproofsFixture {
template<class... Args> static decltype(auto) isParentAndOf(Args&&... args) { return ImplicationSubproofs::isParentAndOf(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) isSubproofGateElement(Args&&... args) { return ImplicationSubproofs::isSubproofGateElement(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) parkImplicationSubproof(Owner& target, Args&&... args) { return target.parkImplicationSubproof(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) rekeySubproofPark(Owner& target, Args&&... args) { return target.rekeySubproofPark(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) revisitRejectedSubproof(Owner& target, Args&&... args) { return target.revisitRejectedSubproof(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) scopeSubtreeStatementCount(Owner& target, Args&&... args) { return target.scopeSubtreeStatementCount(std::forward<Args>(args)...); }
};
}
