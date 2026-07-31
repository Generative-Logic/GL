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

/// @file
/// @brief Regression test for `ExpressionAnalyzer::cleanUpExpressions`'s
///        retention contract (I-58 / D-93): a class-canonicalization drop
///        removes a statement from the runtime registries ONLY. Its
///        `intKnownStatements` registration and its `intStatementLevelsMap`
///        row are immortal — the registered bit makes every registration
///        path skip row re-creation, and the negated-equality expansion's
///        emit gate reads the levels row as its permanent already-emitted
///        memory. An erased row with a surviving registered bit makes the
///        statement re-emittable but never re-registrable, and the class
///        expansion regenerates sibling variants without bound.

#include "test_harness.hpp"

#include "../prover.hpp"

// Two distinct statements under the SAME validity, one kept and one dropped
// by the class-canonical filter. The dropped statement must leave the
// statement registry, but BOTH levels rows must survive the cleanup.
TEST(cleanup_levels_retention, dropped_statement_keeps_its_levels_row) {
    gl::ExpressionAnalyzer analyzer(std::string("Peano"));
    gl::Memory memory;
    const gl::NameId mainId = memory.nameMap.encode("main");

    // One equivalence class at the cleanup validity with two IntLev members;
    // canonical is the decoded-lex first (int_lev_1_1), so a statement
    // carrying int_lev_1_2 holds a non-canonical member and is dropped.
    gl::EquivalenceClass cls;
    cls.setMembersFromNames({ "int_lev_1_1", "int_lev_1_2" }, memory.nameMap);
    const std::vector<char> blob = gl::serializeEquivalenceClass(cls);
    const int32_t lens[1] = { static_cast<int32_t>(blob.size()) };
    memory.equivalenceClassesMap.inner().assignRun(mainId, blob.data(), lens, 1);

    const gl::EncodedExpression keptSrc(
        "(=[a,int_lev_1_1])", memory.nameMap.decode(mainId));
    const gl::EncodedExpression droppedSrc(
        "(=[a,int_lev_1_2])", memory.nameMap.decode(mainId));
    const gl::IntEncodedExpr kept =
        gl::encodeExpression(keptSrc, memory.nameMap);
    const gl::IntEncodedExpr dropped =
        gl::encodeExpression(droppedSrc, memory.nameMap);
    ASSERT_EQ(kept.validityId, dropped.validityId);
    ASSERT_NE(kept.originalId, dropped.originalId);

    memory.intEncodedStatements.push_back(kept);
    memory.intEncodedStatements.push_back(dropped);

    const int64_t keptKey =
        gl::packStatementKey(kept.originalId, kept.validityId);
    const int64_t droppedKey =
        gl::packStatementKey(dropped.originalId, dropped.validityId);
    memory.intStatementLevelsMap.insertSorted(keptKey, 3);
    memory.intStatementLevelsMap.insertSorted(droppedKey, 5);

    analyzer.cleanUpExpressions(memory, gl::StrSpan("main", 4));

    // Registry: only the kept statement remains.
    ASSERT_EQ(static_cast<int>(memory.intEncodedStatements.size()), 1);
    ASSERT_EQ(memory.intEncodedStatements[0].originalId, kept.originalId);

    // Levels map: BOTH rows survive — the dropped statement's row is the
    // immortal already-emitted memory, not runtime state.
    ASSERT_NE(memory.intStatementLevelsMap.lookup(keptKey), 0);
    ASSERT_NE(memory.intStatementLevelsMap.lookup(droppedKey), 0);
}
