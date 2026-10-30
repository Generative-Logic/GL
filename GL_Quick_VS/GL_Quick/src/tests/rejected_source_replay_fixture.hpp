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
#include "../logical_block/rejected_source_replay.hpp"
#include <utility>
namespace gl::tests {
struct RejectedSourceReplayFixture {
static void emitIntegrationRevivalToInternalMailIn(RejectedSourceReplay& target,
 const std::string& primaryConstituentPre, const std::string& primaryConstituentPost,
 const std::vector<std::string>& siblingsPre, const std::vector<std::string>& siblingsPost,
 const std::set<int>& levels, const std::string& validityName,
 const std::set<std::string>& equalitiesForOrigin, Memory& mb) {
 target.emitIntegrationRevivalToInternalMailIn(primaryConstituentPre, primaryConstituentPost,
  siblingsPre, siblingsPost, levels, validityName, equalitiesForOrigin, mb);
}
static void emitIntegrationRevivalToInternalMailIn(RejectedSourceReplay& target,
 StrSpan primaryPre, StrSpan primaryPost, const StrSpan* siblingsPre, int32_t siblingsPreN,
 const StrSpan* siblingsPost, int32_t siblingsPostN, const int* levels, int32_t levelCount,
 StrSpan validityName, const StrSpan* equalitiesForOrigin, int32_t equalityN, Memory& mb) {
 target.emitIntegrationRevivalToInternalMailIn(primaryPre, primaryPost, siblingsPre,
  siblingsPreN, siblingsPost, siblingsPostN, levels, levelCount, validityName,
  equalitiesForOrigin, equalityN, mb);
}
template<class Owner, class... Args> static decltype(auto) hasPotentialParkedDemand(Owner& target, Args&&... args) { return target.hasPotentialParkedDemand(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) queueCanonicalWake(Owner& target, Args&&... args) { return target.queueCanonicalWake(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) queueParkedExistence(Owner& target, Args&&... args) { return target.queueParkedExistence(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) recordZoomInOrigin(Owner& target, Args&&... args) { return target.recordZoomInOrigin(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) releaseAncestorRejected2(Owner& target, Args&&... args) { return target.releaseAncestorRejected2(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) releaseAncestorRejectedIntegration2(Owner& target, Args&&... args) { return target.releaseAncestorRejectedIntegration2(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) releaseParkedProductIntoScope(Owner& target, Args&&... args) { return target.releaseParkedProductIntoScope(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) resetParkedOrStatementRegistries(Owner& target, Args&&... args) { return target.resetParkedOrStatementRegistries(std::forward<Args>(args)...); }
};
}
