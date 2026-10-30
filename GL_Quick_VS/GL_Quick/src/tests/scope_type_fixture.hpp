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
#include "../logical_block/scope_type_resolver.hpp"
#include <utility>
namespace gl::tests {
struct ScopeTypeFixture {
template <typename... Args> static decltype(auto) appendVarDefSetPair(Args&&... args) { return ScopeTypeResolver::appendVarDefSetPair(std::forward<Args>(args)...); }
template <typename Owner, typename... Args> static decltype(auto) buildLbVarDefinitionSetsRecord(Owner& target, Args&&... args) { return target.buildLbVarDefinitionSetsRecord(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) chainVarLabel(Args&&... args) { return ScopeTypeResolver::chainVarLabel(std::forward<Args>(args)...); }
template <typename Owner, typename... Args> static decltype(auto) elementVarLabel(Owner& target, Args&&... args) { return target.elementVarLabel(std::forward<Args>(args)...); }
template <typename Owner, typename... Args> static decltype(auto) equalityStartLabel(Owner& target, Args&&... args) { return target.equalityStartLabel(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) goalAtScope(Args&&... args) { return ScopeTypeResolver::goalAtScope(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) isFlatOperatorLeaf(Args&&... args) { return ScopeTypeResolver::isFlatOperatorLeaf(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) leadingExpressionOfKey(Args&&... args) { return ScopeTypeResolver::leadingExpressionOfKey(std::forward<Args>(args)...); }
template <typename... Args> static decltype(auto) lookupLbVarDefinitionSet(Args&&... args) { return ScopeTypeResolver::lookupLbVarDefinitionSet(std::forward<Args>(args)...); }
template <typename Owner, typename... Args> static decltype(auto) resolvedLabel(Owner& target, Args&&... args) { return target.resolvedLabel(std::forward<Args>(args)...); }
template <typename Owner, typename... Args> static decltype(auto) scopeGoalArgLabel(Owner& target, Args&&... args) { return target.scopeGoalArgLabel(std::forward<Args>(args)...); }
template <typename Owner, typename... Args> static decltype(auto) scopeVarLabel(Owner& target, Args&&... args) { return target.scopeVarLabel(std::forward<Args>(args)...); }
template <typename Owner, typename... Args> static decltype(auto) typeLeafInto(Owner& target, Args&&... args) { return target.typeLeafInto(std::forward<Args>(args)...); }
};
}
