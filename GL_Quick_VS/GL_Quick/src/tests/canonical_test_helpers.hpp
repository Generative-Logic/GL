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
#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "prover_fixture.hpp"
#include "../prover.hpp"

// Direct hook tests supply class membership without the equality ingestion path.
// Give each assumed pair the history that this path would have registered.
inline void seedFixtureClassEvidence(gl::Memory& memory,const gl::EquivalenceClass& cls,gl::StrSpan scope) {
    const std::string validity(scope.ptr,scope.len);
    for(auto from:cls.memberIds) for(auto to:cls.memberIds) if(from!=to) {
        const std::string equality="(=["+((memory).nameMap).decode(from)+","+((memory).nameMap).decode(to)+"])";
        gl::addOriginEncoded(((memory).exprOriginMap),((memory).originInterner),gl::StrSpan(equality),
            gl::StrSpan(validity),gl::OriginTag::broadcast,nullptr,0,8);
    }
}

// Raw class fixtures bypass the production equality-update seam. Prepare the
// second store explicitly without re-keying the ledgers a test is about to inspect.
inline void seedFixtureCanonicalClasses(gl::Memory& memory) {
    gl::tests::ProverFixture analyzer("Peano");
    analyzer.parameters.trackHistory = false;
    std::set<gl::NameId> scopes{gl::NameMap::MAIN_ID};
    for (int32_t i = 1; i <= ((memory).nameMap).nameCount(); ++i)
        if (((memory).nameMap).parentOf(i) != 0) scopes.insert(i);
    for (int32_t i = 1; i <= ((memory).equivalenceClassesMap).count(); ++i)
        scopes.insert(((memory).equivalenceClassesMap).keyAt(i));
    std::vector<gl::NameId> ordered(scopes.begin(), scopes.end());
    std::sort(ordered.begin(), ordered.end(), [&](gl::NameId a, gl::NameId b) {
        const auto da = ((memory).nameMap).ancLen(a), db = ((memory).nameMap).ancLen(b);
        return da != db ? da < db : ((memory).nameMap).decode(a) < ((memory).nameMap).decode(b);
    });
    for (auto scope : ordered) {
        analyzer.canonicalizer.initializeCanonicalScope(memory, scope);
        analyzer.canonicalizer.rebuildCanonicalScope(memory, scope);
    }
}

// Step-4b expression cleanup over explicit scopes, swept in the given order.
inline void cleanUpScopes(gl::tests::ProverFixture& analyzer, gl::Memory& memory,
                          std::initializer_list<gl::NameId> scopeIds) {
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::ColdHashSet<gl::PodKeyStore<gl::NameId>> scopes(
        &gl::genScratchArenas().forSlot(0), &dirty);
    scopes.resetToFresh();
    for (const gl::NameId scope : scopeIds) scopes.mint(scope);
    analyzer.equalityPropagation.cleanUpExpressions(memory, scopes);
}
