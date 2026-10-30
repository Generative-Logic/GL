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
#include "test_harness.hpp"
#include "isolated_test_directory.hpp"
#include "prover_fixture.hpp"
#include "proof_reconstructor_fixture.hpp"
#include "../prover.hpp"
#include "../project_paths.hpp"
#include <fstream>
#include <iterator>
#include <chrono>

TEST(export_components, constructs_explicit_operation_interfaces) {
    gl::tests::ProverFixture engine(std::string("Peano"));
    const auto initialTheoremCount = engine.globalTheoremList.size();
    [[maybe_unused]] gl::ProofReconstructor proofReconstructor{engine.simpleMapStore, engine.exportTheoremUsage, engine.exportCurrentChapterTheorem};
    [[maybe_unused]] gl::ProofGraphWriter proofGraphWriter{engine.body, engine.globalTheoremList, engine.globalTheoremStrings, engine.globalTheoremProducers, engine.definitions, engine.simpleMapStore, engine.cachedProofStacks, engine.exportTheoremUsage, engine.exportCurrentChapterTheorem, engine.orElimCitedOrByMerged, engine.compilationPersistence, engine.theoremPublisher, engine.proofReconstructor};
    ASSERT_EQ(engine.globalTheoremList.size(), initialTheoremCount);
}

TEST(export_components, reconstruction_state_boundaries) {
    using R = gl::ProofReconstructor;
    using F = gl::tests::ProofReconstructorFixture;
    const gl::ExpressionWithValidity goal("(goal)", "main");
    R::clearBuildStackPath();
    ASSERT_FALSE(F::pathContains(goal));
    R::enterChapterGoal(goal);
    ASSERT_TRUE(F::pathContains(goal));
    F::seedTraversalState(goal);
    ASSERT_FALSE(F::traversalStateEmpty());
    R::beginRootTraversal();
    ASSERT_TRUE(F::traversalStateEmpty());
    ASSERT_TRUE(F::pathContains(goal));
    R::leaveChapterGoal(goal);
    ASSERT_FALSE(F::pathContains(goal));
    R::enterChapterGoal(goal);
    F::seedTraversalState(goal);
    R::clearBuildStackPath();
    ASSERT_FALSE(F::pathContains(goal));
    ASSERT_TRUE(F::traversalStateEmpty());
}

TEST(export_components, sorts_expression_weights_and_rejects_mismatch) {
    gl::tests::ProverFixture engine("Peano");
    const std::vector<gl::ExpressionWithValidity> values = {
        {"(a)", "main"}, {"(b)", "main"}, {"(c)", "main"}};
    const auto ordered = engine.proofReconstructor.sortByValuesDesc(values, {2, 5, 1});
    ASSERT_EQ(ordered[0].original, std::string("(b)"));
    ASSERT_EQ(ordered[1].original, std::string("(a)"));
    ASSERT_EQ(ordered[2].original, std::string("(c)"));
    bool rejected = false;
    try { engine.proofReconstructor.sortByValuesDesc(values, {1}); }
    catch (const std::invalid_argument&) { rejected = true; }
    ASSERT_TRUE(rejected);
}

TEST(export_components, writes_synthetic_chapter_and_empty_debug_mapping) {
    const auto selected = gl::tests::testEnvironment("GL_UNIT_TEST_ONLY");
    const auto isolatedRoot = gl::tests::testEnvironment("GL_ISOLATED_EXPORT_ROOT");
    if (!isolatedRoot) {
        gl::tests::runInIsolatedProject("export_components.writes_synthetic_chapter_and_empty_debug_mapping");
        return;
    }
    ASSERT_TRUE(std::filesystem::equivalent(*isolatedRoot, gl::projectRoot()));
    ASSERT_TRUE(selected.has_value());
    ASSERT_EQ(*selected, std::string("export_components.writes_synthetic_chapter_and_empty_debug_mapping"));
    gl::tests::ProverFixture engine("Peano");
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto directory = std::filesystem::path(".debug") /
        ("export_component_" + std::to_string(stamp));
    engine.proofGraphWriter.findEnds({}, directory / "debug");
    ASSERT_TRUE(std::filesystem::exists(directory / "debug" / "global_theorem_list.txt"));
    ASSERT_TRUE(engine.globalTheoremList.empty());
    const std::vector<std::tuple<std::string, std::string, std::string, std::string>> rows = {
        {"(T)", "reformulated statement", "(U)", ""}};
    // Both compiler-state output and chapter output belong to this child tree.
    engine.anchorID_ = "UnitTestExport_" + std::to_string(stamp);
    engine.proofGraphWriter.generateRawProofGraph(rows, directory / "chapters");
    std::ifstream chapter(directory / "chapters" / "0_reformulated_statement.txt");
    ASSERT_TRUE(chapter.is_open());
    const std::string bytes{std::istreambuf_iterator<char>(chapter), std::istreambuf_iterator<char>()};
    ASSERT_EQ(bytes, std::string("(T)\tmain\treformulated from\t(U)\tmain\n"));
    ASSERT_TRUE(engine.exportTheoremUsage.reaches("(T)", "(U)"));
    ASSERT_FALSE(engine.exportTheoremUsage.reaches("(U)", "(T)"));
}

TEST(export_components, vacuous_truth_history_at_main_marks_the_closing_block) {
    gl::tests::ProverFixture engine("Peano");
    gl::Memory& mb = engine.body;
    const int maxOrigins = engine.parameters.max_origin_per_expr;
    ASSERT_TRUE(maxOrigins >= 2);
    const std::string goal("(=[vt_goal,vt_zero])");
    const std::string other("(=[vt_other,vt_zero])");
    const gl::StrSpan main("main", 4);
    const gl::OriginDep deps[3] = {
        { gl::StrSpan("(=[vt_n,vt_zero])", 17), main },
        { gl::StrSpan("!(=[vt_n,vt_zero])", 18), main },
        { gl::StrSpan("(=[vt_n,vt_zero])", 17), main } };
    using W = gl::ProofGraphWriter;
    ASSERT_FALSE(W::hasVacuousTruthAtMain(mb, goal));
    // A vacuous-truth line below main does not mark the block.
    gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner, gl::StrSpan(goal),
        gl::StrSpan("main_boundary_vt_scope", 22), gl::OriginTag::vacuousTruth, deps, 3, maxOrigins);
    ASSERT_FALSE(W::hasVacuousTruthAtMain(mb, goal));
    gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner, gl::StrSpan(goal), main,
        gl::OriginTag::vacuousTruth, deps, 3, maxOrigins);
    ASSERT_TRUE(W::hasVacuousTruthAtMain(mb, goal));
    // Another tag at main does not mark it; a vacuous-truth record after it does.
    gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner, gl::StrSpan(other), main,
        gl::OriginTag::broadcast, nullptr, 0, maxOrigins);
    ASSERT_FALSE(W::hasVacuousTruthAtMain(mb, other));
    gl::addOriginEncoded(mb.exprOriginMap, mb.originInterner, gl::StrSpan(other), main,
        gl::OriginTag::vacuousTruth, deps, 3, maxOrigins);
    ASSERT_TRUE(W::hasVacuousTruthAtMain(mb, other));
}
