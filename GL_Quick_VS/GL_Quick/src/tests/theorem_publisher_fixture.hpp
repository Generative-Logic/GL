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
#include "../proof/theorem_publisher.hpp"
#include <utility>
namespace gl::tests {
struct TheoremPublisherFixture {
template<class Owner, class... Args> static decltype(auto) accessMemory(Owner& target, Args&&... args) { return target.accessMemory(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) compareProducerChains(Args&&... args) { return TheoremPublisher::compareProducerChains(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) reformulateTheorem(Owner& target, Args&&... args) { return target.reformulateTheorem(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) repairTierCitationOrigins(Owner& target, Args&&... args) { return target.repairTierCitationOrigins(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) saveProvedTheorems(Owner& target, Args&&... args) { return target.saveProvedTheorems(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) tryBackReformulateOperatorHead(Owner& target, Args&&... args) { return target.tryBackReformulateOperatorHead(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) updateGlobalSingleChapter(Owner& target, Args&&... args) { return target.updateGlobalSingleChapter(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) updateGlobalSingleChapterLess(Args&&... args) { return TheoremPublisher::updateGlobalSingleChapterLess(std::forward<Args>(args)...); }
};
}
