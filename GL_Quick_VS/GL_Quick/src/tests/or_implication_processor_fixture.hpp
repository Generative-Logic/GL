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
#include "../logical_block/or_implication_processor.hpp"
#include <utility>
namespace gl::tests {
struct OrImplicationProcessorFixture {
static int32_t bindOrCanonicalView(OrImplicationProcessor& target, Memory& mb, OrImplicationMatchIndex& index, OrViewIdentity identity, NameId originalId) { return target.bindOrCanonicalView(mb, index, identity, originalId); }
template<class Owner, class... Args> static decltype(auto) bindOrSourceViews(Owner& target, Args&&... args) { return target.bindOrSourceViews(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) buildOrClassDirectory(Owner& target, Args&&... args) { return target.buildOrClassDirectory(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) ensureOrCanonicalCache(Owner& target, Args&&... args) { return target.ensureOrCanonicalCache(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) evaluateOrImplication(Owner& target, Args&&... args) { return target.evaluateOrImplication(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) materializeOrImplication(Owner& target, Args&&... args) { return target.materializeOrImplication(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) prepareOrImplicationMatches(Owner& target, Args&&... args) { return target.prepareOrImplicationMatches(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) projectOrCacheViews(Owner& target, Args&&... args) { return target.projectOrCacheViews(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) projectOrUser(Owner& target, Args&&... args) { return target.projectOrUser(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) canonicalOrFormId(Owner& target, Args&&... args) { return target.canonicalOrFormId(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) rewriteLocalAt(Owner& target, Args&&... args) { return target.rewriteLocalAt(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) postOrViewTokens(Owner& target, Args&&... args) { return target.postOrViewTokens(std::forward<Args>(args)...); }
static void publishOrCanonicalCache(OrImplicationProcessor& target, Memory& mb, OrImplicationMatchIndex& index, int64_t key, OrCanonicalCache value) { target.publishOrCanonicalCache(mb, index, key, value); }
template<class Owner, class... Args> static decltype(auto) refreshOrCanonicalViews(Owner& target, Args&&... args) { return target.refreshOrCanonicalViews(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) retireDuplicateOrSources(Owner& target, Args&&... args) { return target.retireDuplicateOrSources(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) retireOrImplicationSource(Owner& target, Args&&... args) { return target.retireOrImplicationSource(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) selectOrImplication(Owner& target, Args&&... args) { return target.selectOrImplication(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) stageOrClassMember(Owner& target, Args&&... args) { return target.stageOrClassMember(std::forward<Args>(args)...); }
};
}
