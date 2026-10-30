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
#include "../inference/hash_rule_index.hpp"
#include <cstring>

namespace gl::tests {

// Test scaffolding for the premise-pattern request filter
// (`premisePatternAdmits`): tests that hand-build key maps instead of
// installing rules give their rows matching premise patterns, so the key maps
// alone decide what the filter keeps.

// Encode an expression and place it at a scope.
inline IntEncodedExpr premiseRowAt(NameMap& nm, const char* text, NameId scope) {
    IntEncodedExpr row = encodeExpression(
        StrSpan(text, static_cast<int32_t>(std::strlen(text))), StrSpan("main", 4), nm);
    row.validityId = scope;
    return row;
}

// Register one premise directly (the shape's u_ mask and the pattern's
// scope-major owner) -- the state a staged install leaves after its flush.
inline void registerPremisePattern(HashMemory& hm, const IntEncodedExpr& premise,
                                   NameId scope, int32_t copyId) {
    ScratchArena& g = genScratchArenas().forSlot(genScratchArenas().slotCount() - 1);
    char key[kMaxPremisePatternKeyBytes];
    const int64_t mask = premiseUnchangeableMask(premise);
    int32_t len = premiseShapeKeyInto(premise, key, kMaxPremisePatternKeyBytes);
    HashRuleIndex::addOwnerToRun(hm.premisePatterns, StrSpan(key, len), mask, g);
    len = premisePatternKeyInto(premise, mask, key, kMaxPremisePatternKeyBytes);
    HashRuleIndex::addOwnerToRun(hm.premisePatterns, StrSpan(key, len),
        packPremisePatternOwner(scope, copyId), g);
}

// Give every stored row a live premise of its own shape at its own scope in
// the overall hash memory, so the premise-pattern condition admits every row.
inline void admitEveryStatementShape(Memory& m) {
    for (int32_t i = 0; i < m.intEncodedStatements.size(); ++i) {
        const IntEncodedExpr row = m.intEncodedStatements[i];
        registerPremisePattern(m.overallHashMemory, row, row.validityId, 1);
    }
}

} // namespace gl::tests
