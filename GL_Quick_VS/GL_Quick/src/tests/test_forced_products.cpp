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
/// @brief Unit tests for the forced-product list of a logic block
///        (D-522).
///
/// @details
/// The deposit door lists every statement a forced disintegration (a status-0
/// seed or a hypothesis shot) produces, whatever names it carries and whether
/// or not the block knew it before, and never the disintegrated statement
/// itself; an ordinary disintegration lists nothing. A routing arrival
/// carrying the from-premise mail flag is listed, also when the block knew it
/// before. The matcher, the walk end point and the gate cases live beside the
/// witness-history tests (`test_witness_history_gate.cpp`).

#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"
#include "../prover.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace {
    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief A block with the one-witness existence `existence96` (witness
    ///        `w` with `in[w,u_1]` and the operator fact `in3[u_2,u_3,w,u_4]`)
    ///        and the conjunction `and91` (`in[u_1,u_2]` and `in[u_3,u_2]`,
    ///        no witness), deposited through the door at `main`.
    struct DoorFixture {
        gl::tests::ProverFixture analyzer{"Peano"};
        gl::Memory memory;
        int levels[2] = {0, 2};

        DoorFixture() {
            gl::CompiledDefinitionFixture::operators(analyzer.definitions).clear();
            gl::CompiledDefinitionFixture::operators(analyzer.definitions).insert("in3");
            gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence96"] =
                gl::LogicalEntity("existence", {"(in[w,u_1])", "(in3[u_2,u_3,w,u_4])"},
                    "(existence96[u_1,u_2,u_3,u_4])", 4);
            gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["and91"] =
                gl::LogicalEntity("and", {"(in[u_1,u_2])", "(in[u_3,u_2])"}, "(and91[u_1,u_2,u_3])", 3);
            analyzer.supportingRuleCompiler.compileDisintegrationPackages();
            analyzer.parameters.trackHistory = true;
            memory.level = 2;
            memory.nameMap.encode(sp("main"));
        }

        /// @brief Deposit @p expr at `main` with @p status.
        void deposit(const char* expr, int status, bool recursionHead = false) {
            analyzer.statementMemory.addExprToMemoryBlock(sp(expr), memory, 0, status, levels, 2,
                gl::TransientOrigin{true, gl::OriginTag::taskFormulation, nullptr, 0},
                -1, -1, sp("main"), false, false, nullptr, recursionHead);
        }

        /// @brief The listed forced products as `expression@scope`.
        std::vector<std::string> listed() {
            std::vector<std::string> out;
            const auto& products = memory.lbMemory.forcedProducts;
            for (int32_t id = 1; id <= products.count(); ++id) {
                const gl::StatementKey key = gl::Codec<gl::StatementKey>::decode(products.keyAt(id));
                out.push_back(memory.nameMap.decode(key.orig) + "@" + memory.nameMap.decode(key.validity));
            }
            return out;
        }
    };

    /// @brief Whether @p list holds an entry starting with @p prefix.
    bool hasPrefix(const std::vector<std::string>& list, const std::string& prefix) {
        for (const std::string& s : list)
            if (s.rfind(prefix, 0) == 0) return true;
        return false;
    }

    /// @brief Whether @p list holds @p entry.
    bool has(const std::vector<std::string>& list, const std::string& entry) {
        for (const std::string& s : list)
            if (s == entry) return true;
        return false;
    }
} // namespace

// A status-0 seed's disintegration lists its products, the forced witness's
// operator fact among them, and not the seed itself.
TEST(forced_products, door_lists_the_products_of_a_forced_disintegration) {
    DoorFixture f;
    f.deposit("(existence96[1,2,3,4])", 0);
    const std::vector<std::string> listed = f.listed();
    ASSERT_FALSE(listed.empty());
    ASSERT_TRUE(hasPrefix(listed, "(in3[2,3,force_lev_2_"));
    ASSERT_FALSE(has(listed, "(existence96[1,2,3,4])@main"));
}

// The split of a forced conjunction is listed although its parts carry no
// forced name.
TEST(forced_products, door_lists_a_forced_split_without_forced_names) {
    DoorFixture f;
    f.deposit("(and91[a,N,c])", 0);
    const std::vector<std::string> listed = f.listed();
    ASSERT_TRUE(has(listed, "(in[a,N])@main"));
    ASSERT_TRUE(has(listed, "(in[c,N])@main"));
    ASSERT_FALSE(has(listed, "(and91[a,N,c])@main"));
}

// A forced product the block knew before is listed too: the premise gives it
// whatever derived it first.
TEST(forced_products, door_lists_a_forced_product_known_before) {
    DoorFixture f;
    f.deposit("(in[a,N])", 1);
    ASSERT_TRUE(f.listed().empty());
    f.deposit("(and91[a,N,c])", 0);
    const std::vector<std::string> listed = f.listed();
    ASSERT_TRUE(has(listed, "(in[a,N])@main"));
    ASSERT_TRUE(has(listed, "(in[c,N])@main"));
}

// An ordinary disintegration lists nothing.
TEST(forced_products, door_lists_nothing_for_an_ordinary_disintegration) {
    DoorFixture f;
    f.deposit("(existence96[1,2,3,4])", 1);
    f.deposit("(and91[a,N,c])", 1);
    ASSERT_TRUE(f.listed().empty());
}

// The conclusion of a hypothesis shot is force-disintegrated, so its
// products are listed too.
TEST(forced_products, door_lists_the_products_of_a_hypothesis_shot) {
    DoorFixture f;
    f.memory.recursionAuxyImplId = 1;
    f.deposit("(existence96[1,2,3,4])", 1, true);
    const std::vector<std::string> listed = f.listed();
    ASSERT_TRUE(hasPrefix(listed, "(in3[2,3,force_lev_2_"));
    ASSERT_FALSE(has(listed, "(existence96[1,2,3,4])@main"));
}

// A routing arrival with the from-premise flag is listed, also when the block
// knew the statement before; the same arrival without the flag is not.
TEST(forced_products, absorb_lists_an_arrival_marked_from_premise) {
    for (int mode = 0; mode < 3; ++mode) {
        const bool fromPremise = mode != 1;
        const bool knownBefore = mode == 2;
        gl::tests::ProverFixture analyzer{"Peano"};
        gl::Memory memory;
        analyzer.parameters.trackHistory = true;
        analyzer.parameters.max_origin_per_expr = 16;
        memory.isActive = true;
        memory.level = 1;
        memory.parentMemory = &analyzer.body;
        ASSERT_EQ(memory.nameMap.encode(sp("main")), gl::NameMap::MAIN_ID);
        if (knownBefore) {
            const int ownLevels[] = {0};
            analyzer.statementMemory.addExprToMemoryBlock(sp("(in[a,N])"), memory, 0, 1, ownLevels, 1,
                gl::TransientOrigin{true, gl::OriginTag::taskFormulation, nullptr, 0},
                -1, -1, sp("main"), false);
            ASSERT_EQ(memory.lbMemory.forcedProducts.count(), 0);
        }
        const int32_t original = gl::mailInterner().mint(sp("(in[a,N])"));
        const int32_t validity = gl::mailInterner().mint(sp("main"));
        const int levels[] = {0};
        memory.mailIn.insertStatement(original, validity, levels, 1);
        if (fromPremise)
            memory.mailIn.mergeStatementFlag(gl::packOriginKey(original, validity),
                gl::packMailStatementFlag(gl::kMailStatementFlagNone, -1, true));
        gl::addMailOriginRecord(memory.mailIn.origins_, gl::packOriginKey(original, validity),
            static_cast<uint8_t>(gl::OriginTag::taskFormulation), nullptr, 0, 16);
        analyzer.logicBlockOperations.standardProcessing(memory, &memory.mailIn,
            memory.sameIterationInternalMail, memory.nextIterationInternalMail, -1);
        memory.mailIn.clear();
        const gl::NameId expr = memory.nameMap.lookup(sp("(in[a,N])"));
        ASSERT_TRUE(expr != 0);
        const bool listed = memory.lbMemory.forcedProducts.lookup(
            gl::packStatementKey(expr, gl::NameMap::MAIN_ID)) != 0;
        ASSERT_EQ(listed, fromPremise);
    }
}
