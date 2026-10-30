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
#include "../logical_block/watch_processor.hpp"
#include <utility>
namespace gl::tests {
struct WatchProcessorFixture {
template<class Owner, class... Args> static decltype(auto) bookOrEqualitySupport(Owner& target, Args&&... args) { return target.bookOrEqualitySupport(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) canonicalFormCached(Owner& target, Args&&... args) { return target.canonicalFormCached(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) canonicalIdCached(Owner& target, Args&&... args) { return target.canonicalIdCached(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) canonicalRepresentativeAt(Args&&... args) { return WatchProcessor::canonicalRepresentativeAt(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) canonicalTextCached(Owner& target, Args&&... args) { return target.canonicalTextCached(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) canonicalWatchIdAt(Owner& target, Args&&... args) { return target.canonicalWatchIdAt(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) clearWatch(Owner& target, Args&&... args) { return target.clearWatch(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) ensureCanonicalStatementUses(Owner& target, Args&&... args) { return target.ensureCanonicalStatementUses(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) ensureCanonicalWatchUses(Owner& target, Args&&... args) { return target.ensureCanonicalWatchUses(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) ensureGoalLedgerCurrent(Owner& target, Args&&... args) { return target.ensureGoalLedgerCurrent(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) importNegationIntoDescendantClassScopes(Owner& target, Args&&... args) { return target.importNegationIntoDescendantClassScopes(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) recordWatch(Owner& target, Args&&... args) { return target.recordWatch(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) recordWatchCanonical(Owner& target, Args&&... args) { return target.recordWatchCanonical(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) sameCanonicalSpelling(Args&&... args) { return WatchProcessor::sameCanonicalSpelling(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) watchKey(Args&&... args) { return WatchProcessor::watchKey(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) watchRunContainsValue(Args&&... args) { return WatchProcessor::watchRunContainsValue(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) watchRunHasCanonical(Args&&... args) { return WatchProcessor::watchRunHasCanonical(std::forward<Args>(args)...); }
template<class... Args> static decltype(auto) watchShaped(Args&&... args) { return WatchProcessor::watchShaped(std::forward<Args>(args)...); }
template<class Owner, class... Args> static decltype(auto) writeLiteralTwin(Owner& target, Args&&... args) { return target.writeLiteralTwin(std::forward<Args>(args)...); }
};
}
