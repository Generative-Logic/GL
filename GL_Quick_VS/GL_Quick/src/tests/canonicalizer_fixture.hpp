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
#include "../logical_block/canonicalizer.hpp"
#include <utility>
namespace gl::tests {
struct CanonicalizerFixture {
template <typename... Args> static decltype(auto) appliedEqualityIsLocal(Canonicalizer& target, Args&&... args) { return target.appliedEqualityIsLocal(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) equalityScopeChain(Canonicalizer& target, Args&&... args) { return target.equalityScopeChain(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) equalityRowVerdict(Canonicalizer& target, Args&&... args) { return target.equalityRowVerdict(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) directEqualityVerdict(Canonicalizer& target, Args&&... args) { return target.directEqualityVerdict(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) classHasLocalEqualityRow(Canonicalizer& target, Args&&... args) { return target.classHasLocalEqualityRow(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) canonicalClassContaining(Canonicalizer& target, Args&&... args) { return target.canonicalClassContaining(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) liveCanonicalMemberAt(Canonicalizer& target, Args&&... args) { return target.liveCanonicalMemberAt(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) resolveStatementTokens(Canonicalizer& target, Args&&... args) { return target.resolveStatementTokens(std::forward<Args>(args)...); }
};
}
