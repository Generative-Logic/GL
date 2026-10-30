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
/// @brief Unit tests for the shared shot routine of the disintegration
///        package (I-333).
///
/// @details
/// `shootUniversal` fires a universal once on products: the compiler with
/// placeholders, the disintegrator with temporary witnesses or permanent
/// names. The cases pin the matching rules (operator and polarity, fixed
/// arguments, consistent bound variables, the `u_` spelling ignored), the
/// head instantiation, and the cases that fire nothing. The package cases
/// pin what the compiler stores per existence, the frame cases how the
/// disintegrator instantiates a package on a witness path, and the runtime
/// case I1's existence unfolding its shot heads with their history.

#include "../logical_block/disintegration_shots.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"
#include "../prover.hpp"

#include <map>
#include <set>
#include <string>
#include <vector>

namespace {
    // 1 MiB pool / 256 KiB block.
    const gl::StaticMemoryConfig kShotTestCfg{ 1 << 20, 1 << 18 };

    /// @brief Run `shootUniversal` and return the heads as strings.
    std::vector<std::string> shoot(gl::ScratchArena& arena, const std::string& rule,
        const std::vector<std::string>& products, std::vector<std::vector<int32_t>>* premises = nullptr) {
        std::vector<gl::StrSpan> spans;
        for (const std::string& product : products) spans.emplace_back(product);
        gl::UniversalShot shots[16];
        const int32_t n = gl::shootUniversal(gl::StrSpan(rule), spans.data(),
            static_cast<int32_t>(spans.size()), arena, shots, 16);
        std::vector<std::string> heads;
        for (int32_t i = 0; i < n; ++i) {
            heads.push_back(shots[i].head.toStdString());
            if (premises != nullptr) {
                std::vector<int32_t> chosen;
                for (int32_t p = 0; p < shots[i].premiseN; ++p) chosen.push_back(shots[i].premiseProduct[p]);
                premises->push_back(chosen);
            }
        }
        return heads;
    }
} // namespace

// divisorSet's membership universal fires on p in D (the I1 shot): the head
// takes p, the other membership fact (of another set) fires nothing.
TEST(disintegration_shots, single_premise_fires_on_the_matching_product) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    std::vector<std::vector<int32_t>> premises;
    const std::vector<std::string> heads = shoot(arena,
        "(>[1](in[1,u_D])(preorder[u_N,u_A,u_2,1]))",
        { "(in[u_q,u_E])", "(in[u_p,u_D])" }, &premises);
    ASSERT_EQ(heads.size(), static_cast<size_t>(1));
    ASSERT_EQ(heads[0], std::string("(preorder[u_N,u_A,u_2,u_p])"));
    ASSERT_EQ(premises[0].size(), static_cast<size_t>(1));
    ASSERT_EQ(premises[0][0], 1);
}

// A fixed argument of the premise that differs from the product's fires
// nothing.
TEST(disintegration_shots, fixed_argument_mismatch_fires_nothing) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const std::vector<std::string> heads = shoot(arena,
        "(>[1](in[1,u_D])(preorder[u_N,u_A,u_2,1]))", { "(in[u_q,u_E])" });
    ASSERT_TRUE(heads.empty());
}

// The `u_` spelling is ignored in the comparison: the disintegrator's slot
// witness is a bare token, its enclosing witnesses carry `u_`.
TEST(disintegration_shots, u_prefix_is_ignored_in_the_comparison) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const std::vector<std::string> heads = shoot(arena,
        "(>[1](in[1,u_D])(preorder[u_N,u_A,u_2,1]))", { "(in[p,D])" });
    ASSERT_EQ(heads.size(), static_cast<size_t>(1));
    ASSERT_EQ(heads[0], std::string("(preorder[u_N,u_A,u_2,p])"));
}

// Two premises bind their shared variables consistently; a product that
// breaks the binding is not used.
TEST(disintegration_shots, two_premises_bind_consistently) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    std::vector<std::vector<int32_t>> premises;
    const std::vector<std::string> heads = shoot(arena,
        "(>[1,2](in3[1,2,u_n,u_m])(>[](in[1,u_S])(foo[1,2])))",
        { "(in3[u_a,u_b,u_n,u_m])", "(in[u_b,u_S])", "(in[u_a,u_S])" }, &premises);
    ASSERT_EQ(heads.size(), static_cast<size_t>(1));
    ASSERT_EQ(heads[0], std::string("(foo[u_a,u_b])"));
    ASSERT_EQ(premises[0][0], 0);
    ASSERT_EQ(premises[0][1], 2);
}

// A bound variable that only the head names cannot be taken: no shot.
TEST(disintegration_shots, head_variable_not_taken_fires_nothing) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const std::vector<std::string> heads = shoot(arena,
        "(>[1,2](in[1,u_S])(bar[1,2]))", { "(in[u_a,u_S])" });
    ASSERT_TRUE(heads.empty());
}

// Polarity is part of the shape: a negated premise matches a negated
// product only.
TEST(disintegration_shots, negated_premise_matches_a_negated_product_only) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const std::vector<std::string> heads = shoot(arena,
        "(>[1]!(in[1,u_S])(baz[1]))", { "(in[u_a,u_S])", "!(in[u_b,u_S])" });
    ASSERT_EQ(heads.size(), static_cast<size_t>(1));
    ASSERT_EQ(heads[0], std::string("(baz[u_b])"));
}

// A compound head keeps its own binder; only the universal's variables are
// replaced.
TEST(disintegration_shots, compound_head_keeps_its_own_binder) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const std::vector<std::string> heads = shoot(arena,
        "(>[1](in[1,u_S])!(>[2](in[2,u_N])!(in3[1,2,u_n,u_m])))", { "(in[u_p,u_S])" });
    ASSERT_EQ(heads.size(), static_cast<size_t>(1));
    ASSERT_EQ(heads[0], std::string("!(>[2](in[2,u_N])!(in3[u_p,2,u_n,u_m]))"));
}

// Every matching product is one shot, in product order.
TEST(disintegration_shots, every_matching_product_is_one_shot_in_order) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const std::vector<std::string> heads = shoot(arena,
        "(>[1](in[1,u_D])(preorder[u_N,u_A,1,1]))",
        { "(in[u_p,u_D])", "(in[u_r,u_D])" });
    ASSERT_EQ(heads.size(), static_cast<size_t>(2));
    ASSERT_EQ(heads[0], std::string("(preorder[u_N,u_A,u_p,u_p])"));
    ASSERT_EQ(heads[1], std::string("(preorder[u_N,u_A,u_r,u_r])"));
}

// A compound product never matches a flat premise.
TEST(disintegration_shots, compound_product_never_matches) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const std::vector<std::string> heads = shoot(arena,
        "(>[1](in[1,u_D])(preorder[u_N,u_A,u_2,1]))",
        { "(>[1](in[1,u_D])(in[1,u_N]))" });
    ASSERT_TRUE(heads.empty());
}

// I1's existence (the divisor set D of n with a minimum p): divisorSet's three
// membership compacts fire on p in D and attach to the witness of
// hasMinimum's existence; minimum's own compact stays in hasMinimum's package.
TEST(disintegration_package, divisor_set_existence_gets_the_three_membership_shots) {
    gl::tests::ProverFixture ea("FTA");
    std::string expression = "!(>[D](divisorSet[N,a,m,t,n,D])!(hasMinimum[N,a,D]))";
    expression = ea.structuralCompiler.compileExpression(expression);
    const std::string name = gl::extractExpressionUniversalSpan(gl::StrSpan(expression)).toStdString();
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    const auto& entity = gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions).at(name);
    ASSERT_TRUE(entity.shotsPrepared);
    ASSERT_EQ(entity.shotBinders.size(), std::size_t(2));
    ASSERT_EQ(entity.shotPlaceholders.size(), std::size_t(2));
    ASSERT_EQ(entity.shotBinders[0], entity.signature);
    ASSERT_EQ(entity.shots.size(), std::size_t(3));
    const std::string& p = entity.shotPlaceholders[1];
    const std::string& d = entity.shotPlaceholders[0];
    int lowerBound = 0, upperBounds = 0;
    std::set<std::string> upperOperators;
    for (const auto& shot : entity.shots) {
        ASSERT_EQ(shot.binder, 1);
        ASSERT_EQ(shot.premises.size(), std::size_t(1));
        ASSERT_EQ(shot.premises[0], "(in[" + p + "," + d + "])");
        ASSERT_TRUE(shot.head.rfind("(preorder[", 0) == 0);
        const std::vector<std::string> args = ce::getArgs(shot.head);
        ASSERT_EQ(args.size(), std::size_t(4));
        if (args[3] == p) ++lowerBound;
        if (args[2] == p) { ++upperBounds; upperOperators.insert(args[1]); }
    }
    // 2 <= p, then p <= n over + and p | n over *.
    ASSERT_EQ(lowerBound, 1);
    ASSERT_EQ(upperBounds, 2);
    ASSERT_EQ(upperOperators.size(), std::size_t(2));
}

// hasMinimum's existence carries its own shot: minimum's universal on the
// minimum's membership gives the minimum below itself.
TEST(disintegration_package, has_minimum_gets_its_own_reflexive_shot) {
    gl::tests::ProverFixture ea("FTA");
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    const auto& entity = gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions).at("hasMinimum");
    ASSERT_TRUE(entity.category == "existence");
    ASSERT_TRUE(entity.shotsPrepared);
    ASSERT_EQ(entity.shotBinders.size(), std::size_t(1));
    ASSERT_EQ(entity.shots.size(), std::size_t(1));
    ASSERT_EQ(entity.shots[0].binder, 0);
    const std::vector<std::string> args = ce::getArgs(entity.shots[0].head);
    ASSERT_EQ(args.size(), std::size_t(4));
    ASSERT_EQ(args[2], entity.shotPlaceholders[0]);
    ASSERT_EQ(args[3], entity.shotPlaceholders[0]);
}

// An existence without internal universals gets an empty, prepared package,
// and a second pass changes nothing.
TEST(disintegration_package, existence_without_universals_gets_an_empty_package) {
    gl::tests::ProverFixture ea("FTA");
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    const auto& registry = gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions);
    const auto& preorder = registry.at("preorder");
    ASSERT_TRUE(preorder.category == "existence");
    ASSERT_TRUE(preorder.shotsPrepared);
    ASSERT_TRUE(preorder.shots.empty());
    const std::size_t count = registry.size();
    const auto shots = registry.at("hasMinimum").shots.size();
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    ASSERT_EQ(registry.size(), count);
    ASSERT_EQ(registry.at("hasMinimum").shots.size(), shots);
    for (const auto& [name, entity] : registry)
        if (entity.category == "existence") ASSERT_TRUE(entity.shotsPrepared);
}

namespace {
    /// @brief A two-binder package owner: `existence90[u_1,u_2]` whose
    ///        witness (placeholder `u_pi_lev_0_1`) encloses the nested
    ///        existence `existence91[u_1,u_pi_lev_0_1]` (binder 1, placeholder
    ///        `u_pi_lev_0_2`).
    gl::LogicalEntity frameOwner() {
        gl::LogicalEntity owner("existence", {"(in[pi_lev_0_1,u_1])", "(existence91[u_1,pi_lev_0_1])"},
            "(existence90[u_1,u_2])", 2);
        owner.shotBinders = {"(existence90[u_1,u_2])", "(existence91[u_1,u_pi_lev_0_1])"};
        owner.shotPlaceholders = {"u_pi_lev_0_1", "u_pi_lev_0_2"};
        owner.shotsPrepared = true;
        return owner;
    }
} // namespace

// A frame maps the owner's signature to the instance simultaneously and
// spells a filled binder bare as the current one, u_-prefixed otherwise;
// an unfilled binder keeps its placeholder.
TEST(disintegration_frame, instantiates_texts_on_the_path) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const gl::LogicalEntity owner = frameOwner();
    const std::string instance = "(existence90[u_2,u_1])";
    gl::ShotFrame frame;
    gl::bindShotFrame(frame, owner, gl::StrSpan(instance), nullptr);
    ASSERT_EQ(frame.argN, 2);
    ASSERT_TRUE(frame.outer == nullptr);
    const std::string text = "(foo[u_1,u_2,u_pi_lev_0_1,u_pi_lev_0_2])";
    ASSERT_EQ(gl::instantiateShotText(gl::StrSpan(text), frame, -1, arena).toStdString(),
        std::string("(foo[u_2,u_1,u_pi_lev_0_1,u_pi_lev_0_2])"));
    const std::string w = "w";
    frame.binderToken[0] = gl::StrSpan(w);
    ASSERT_EQ(gl::instantiateShotText(gl::StrSpan(text), frame, 0, arena).toStdString(),
        std::string("(foo[u_2,u_1,w,u_pi_lev_0_2])"));
    ASSERT_EQ(gl::instantiateShotText(gl::StrSpan(text), frame, -1, arena).toStdString(),
        std::string("(foo[u_2,u_1,u_w,u_pi_lev_0_2])"));
    const std::string v = "v";
    frame.binderToken[1] = gl::StrSpan(v);
    ASSERT_EQ(gl::instantiateShotText(gl::StrSpan(text), frame, 1, arena).toStdString(),
        std::string("(foo[u_2,u_1,u_w,v])"));
}

// A nested existence fills its binder only below the enclosing witness its
// template names, and only while the binder is unfilled; binder 0 is never
// matched.
TEST(disintegration_frame, nested_binder_matches_below_its_enclosing_witness) {
    gl::GlobalMemoryManager m;
    m.init(kShotTestCfg);
    gl::ScratchArena arena; arena.bind(&m);
    const gl::LogicalEntity owner = frameOwner();
    const std::string instance = "(existence90[u_7,u_8])";
    gl::ShotFrame frame;
    gl::bindShotFrame(frame, owner, gl::StrSpan(instance), nullptr);
    const std::string below = "(existence91[u_7,u_w])";
    ASSERT_EQ(gl::matchShotBinder(frame, gl::StrSpan(below), arena), -1);
    const std::string w = "w";
    frame.binderToken[0] = gl::StrSpan(w);
    ASSERT_EQ(gl::matchShotBinder(frame, gl::StrSpan(below), arena), 1);
    const std::string elsewhere = "(existence91[u_7,u_v])";
    ASSERT_EQ(gl::matchShotBinder(frame, gl::StrSpan(elsewhere), arena), -1);
    ASSERT_EQ(gl::matchShotBinder(frame, gl::StrSpan(instance), arena), -1);
    const std::string x = "x";
    frame.binderToken[1] = gl::StrSpan(x);
    ASSERT_EQ(gl::matchShotBinder(frame, gl::StrSpan(below), arena), -1);
}

// I1's existence disintegrates with its package: p's constituents gain the
// three membership heads and hasMinimum's reflexive head, the heads unfold in
// the same walk (p | n gives a witness d with p * d = n), and the published
// head p | n carries the implication row of its shot: the membership rule
// and p in D.
TEST(disintegration_package, divisor_set_witness_unfolds_its_shot_heads) {
    gl::tests::ProverFixture ea("FTA");
    const std::string compiled =
        ea.structuralCompiler.compileExpression("!(>[D](divisorSet[N,a,m,t,n,D])!(hasMinimum[N,a,D]))");
    ea.supportingRuleCompiler.compileDisintegrationPackages();
    const std::map<std::string, std::string> values{
        {"N", "1"}, {"a", "4"}, {"m", "5"}, {"t", "7"}, {"n", "9"}};
    for (const std::string& argument : ce::getArgs(compiled)) ASSERT_EQ(values.count(argument), std::size_t(1));
    const std::string source = ce::replaceKeysInString(compiled, values);
    ea.parameters.trackHistory = true;
    gl::Memory memory;
    memory.level = 2;
    memory.nameMap.encode(gl::StrSpan("main", 4));
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    int levels[2] = {0, 2};
    gl::DisintProducts output(&arena);
    ea.disintegrationProcessor.disintegrateExpr2(gl::StrSpan(source), memory, 0, true,
        gl::StrSpan("main", 4), output, true, false, levels, 2, true);
    std::set<std::string> statements;
    output.statements.forEachSorted([&](gl::StrSpan text, gl::StrSpan) { statements.insert(text.toStdString()); });

    std::string minimum;
    for (const std::string& statement : statements)
        if (statement.rfind("(minimum[1,4,", 0) == 0) {
            ASSERT_TRUE(minimum.empty());
            minimum = statement;
        }
    ASSERT_FALSE(minimum.empty());
    const std::vector<std::string> minimumArgs = ce::getArgs(minimum);
    const std::string& D = minimumArgs[2];
    const std::string& p = minimumArgs[3];
    ASSERT_EQ(statements.count("(preorder[1,4,7," + p + "])"), std::size_t(1));
    ASSERT_EQ(statements.count("(preorder[1,4," + p + ",9])"), std::size_t(1));
    ASSERT_EQ(statements.count("(preorder[1,5," + p + ",9])"), std::size_t(1));
    ASSERT_EQ(statements.count("(preorder[1,4," + p + "," + p + "])"), std::size_t(1));
    int cofactors = 0;
    for (const std::string& statement : statements) {
        if (statement.rfind("(in3[" + p + ",", 0) != 0) continue;
        const std::vector<std::string> args = ce::getArgs(statement);
        if (args[2] == "9" && args[3] == "5" && args[1] != p && values.count(args[1]) == 0) ++cofactors;
    }
    ASSERT_EQ(cofactors, 1);

    int64_t key = 0;
    ASSERT_TRUE(gl::lookupOriginKey(memory.originInterner, "(preorder[1,5," + p + ",9])", "main", key));
    const int32_t originId = memory.exprOriginMap.lookup(key);
    ASSERT_NE(originId, 0);
    int implicationRows = 0;
    for (const auto& row : memory.exprOriginMap.recordsAt(originId)) {
        const auto line = gl::decodeOrigin(row, memory.originInterner);
        if (line.first != "implication") continue;
        ASSERT_EQ(line.second.size(), std::size_t(2));
        ASSERT_TRUE(line.second[0].original.rfind("(>[", 0) == 0);
        ASSERT_TRUE(line.second[0].original.find("(in[") != std::string::npos);
        ASSERT_EQ(line.second[1].original, "(in[" + p + "," + D + "])");
        ASSERT_EQ(line.second[1].validityName, std::string("main"));
        ++implicationRows;
    }
    ASSERT_EQ(implicationRows, 1);
}
