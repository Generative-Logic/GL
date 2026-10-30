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
#include "../logical_block/equality_propagation.hpp"
#include <utility>
namespace gl::tests {
struct EqualityPropagationFixture {
template<class Owner, class... Args> static decltype(auto) enumerateEqClassRewrites(Owner& target, Args&&... args) { return target.enumerateEqClassRewrites(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) filterIterationsCore(Owner& target, Args&&... args) { return target.filterIterationsCore(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) forEachNonCanonicalMemberToken(Owner& target, Args&&... args) { return target.forEachNonCanonicalMemberToken(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) forEachNonCanonicalMemberTokenCore(Owner& target, Args&&... args) { return target.forEachNonCanonicalMemberTokenCore(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) isSubsetOf(Args&&... args) { return EqualityPropagation::isSubsetOf(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) overwriteLevels(Args&&... args) { return EqualityPropagation::overwriteLevels(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) selectRelayWitnesses(Args&&... args) { return EqualityPropagation::selectRelayWitnesses(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) visitNormalMemberToken(Owner& target, Args&&... args) { return target.visitNormalMemberToken(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) visitSpecialMemberToken(Owner& target, Args&&... args) { return target.visitSpecialMemberToken(std::forward<Args>(args)...); }
};
}
