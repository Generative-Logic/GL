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
#include "../inference/encoded_expression_normalizer.hpp"
#include <utility>
namespace gl::tests {
struct EncodedExpressionNormalizerFixture {
template<class Owner, class... Args> static decltype(auto) extractUnchangeables(Owner& target, Args&&... args) { return target.extractUnchangeables(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) getAllEncodedArgs(Owner& target, Args&&... args) { return target.getAllEncodedArgs(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) makeIntNormalizedKey(Owner& target, Args&&... args) { return target.makeIntNormalizedKey(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) makeIntNormalizedKeyWithMap(Owner& target, Args&&... args) { return target.makeIntNormalizedKeyWithMap(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) resetUnchangeables(Owner& target, Args&&... args) { return target.resetUnchangeables(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) setUnchangeables(Owner& target, Args&&... args) { return target.setUnchangeables(std::forward<Args>(args)...); }
};
}
