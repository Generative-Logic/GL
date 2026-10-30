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
#include "prover_fixture.hpp"
#include <type_traits>
#include <utility>

namespace {
template<class T, class = void> struct HasPublicBody : std::false_type {};
template<class T> struct HasPublicBody<T, std::void_t<decltype(std::declval<T&>().body)>> : std::true_type {};
template<class T, class = void> struct HasPublicParameters : std::false_type {};
template<class T> struct HasPublicParameters<T, std::void_t<decltype(std::declval<T&>().parameters)>> : std::true_type {};
template<class T, class = void> struct HasPublicDefinitions : std::false_type {};
template<class T> struct HasPublicDefinitions<T, std::void_t<decltype(std::declval<T&>().definitions)>> : std::true_type {};
static_assert(!HasPublicBody<gl::Prover>::value);
static_assert(!HasPublicParameters<gl::Prover>::value);
static_assert(!HasPublicDefinitions<gl::Prover>::value);
static_assert(HasPublicBody<gl::tests::ProverFixture>::value);
static_assert(std::is_same_v<decltype(std::declval<const gl::Prover&>().configuration()), const gl::ProverParameters&>);
}

TEST(prover_interface, operation_views_keep_the_original_owners) {
    gl::tests::ProverFixture fixture("Peano");
    gl::Prover& prover = fixture;
    ASSERT_TRUE(&prover.compilationFiles() == &fixture.compilationPersistence);
    ASSERT_TRUE(&prover.compiledNames() == &fixture.compilationReuse);
    ASSERT_TRUE(&prover.expressionExpansion() == &fixture.definitionExpansion);
    ASSERT_TRUE(&prover.compiler() == &fixture.structuralCompiler);
    ASSERT_TRUE(&prover.results() == &fixture.theoremPublisher);
    ASSERT_TRUE(&prover.orConstruction() == &fixture.orTheoremBuilder);
    ASSERT_TRUE(&prover.gridOperations() == &fixture.proofGridBuilder);
    ASSERT_TRUE(&prover.proofWriter() == &fixture.proofGraphWriter);
    ASSERT_TRUE(&prover.definitionRegistry() == &fixture.definitions);
    ASSERT_TRUE(&prover.types() == &fixture.scopeTypes);
    ASSERT_TRUE(&prover.hashRules() == &fixture.hashRuleInstaller);
    ASSERT_TRUE(&prover.statements() == &fixture.statementMemory);
    ASSERT_TRUE(&prover.configuration() == &fixture.parameters);
    prover.setCompressionMode(true);
    ASSERT_TRUE(prover.configuration().compressor_mode);
    ASSERT_TRUE(prover.configuration().ban_disintegration);
    prover.setCompressionMode(false);
    ASSERT_FALSE(prover.configuration().compressor_mode);
    ASSERT_FALSE(prover.configuration().ban_disintegration);
    prover.setPhase2ShortcutProfile(true);
    ASSERT_TRUE(fixture.phase2ShortcutProfile);
    prover.setPhase2ShortcutProfile(false);
    ASSERT_FALSE(fixture.phase2ShortcutProfile);
    const auto directory = std::filesystem::path(".debug") / "interface_filter";
    prover.setFilteredConjecturesDirectory(directory);
    ASSERT_TRUE(fixture.filteredConjecturesDirectory == directory);
    const std::vector<std::string> survivors = {"(a)", "(b)"};
    prover.setCompressionSurvivors(survivors);
    ASSERT_TRUE(prover.compressionSurvivors() == survivors);
    prover.setCompressionSurvivors({});
    ASSERT_TRUE(prover.compressionSurvivors().empty());
    ASSERT_TRUE(&prover.precompiledExternals() == &fixture.precompiledExternalsExport);
    gl::Memory* block = prover.createCompressionBlock();
    ASSERT_TRUE(block != &fixture.body);
    prover.destroyCompressionBlock(block);
    const auto usage = prover.mailUsage();
    ASSERT_EQ(usage.mailLogBlocks, fixture.mailArena.blocksHeld());
    ASSERT_EQ(usage.routingPeak, fixture.peakRoutingMailInBlocks.load(std::memory_order_relaxed));
    ASSERT_EQ(usage.deloadableMailOutPeakBytes, fixture.peakDeloadableMailOutBytes);
    ASSERT_EQ(usage.routingInFlight, fixture.routingMailInBlocksInFlight.load(std::memory_order_relaxed));
    ASSERT_EQ(usage.dormantBlocks, fixture.dormantLogicBlocksAtGridBuild);
    ASSERT_EQ(usage.rollingHistory, fixture.rollingMailHistoryEnabled);
    ASSERT_EQ(usage.committedHistoryBytes, fixture.mailLog.totalCommittedHistoryBytes);
    ASSERT_EQ(usage.retainedHistoryPeakBytes, fixture.mailLog.peakRetainedHistoryBytes);
}

TEST(prover_interface, result_operations_preserve_postprocessing_policy) {
    gl::tests::ProverFixture fixture("Peano");
    gl::Prover& prover = fixture;
    fixture.globalTheoremList = {
        {"(tier)", "proved not broadcast", "", ""},
        {"(merge)", "single_chapter_proof", "", ""},
        {"(drop)", "single_chapter_proof", "", ""},
        {"(keep)", "single_chapter_proof", "", ""}};
    auto& results = prover.results();
    results.sortPublished();
    ASSERT_TRUE(&results.sortedForProofGraph() == &results.published());
    ASSERT_EQ(std::get<0>(results.published().front()), std::string("(drop)"));
    results.archive(results.published().back());
    ASSERT_EQ(results.archived().size(), size_t(1));
    ASSERT_EQ(std::get<0>(results.archived().front()), std::string("(tier)"));
    results.archive(results.published().front());
    ASSERT_TRUE(&results.sortedForProofGraph() == &results.archived());
    ASSERT_EQ(results.archived().size(), size_t(2));
    ASSERT_EQ(std::get<0>(results.archived().front()), std::string("(drop)"));
    results.retainCompressionSurvivors({"(keep)"}, {"(merge)"});
    ASSERT_EQ(results.published().size(), size_t(3));
    ASSERT_EQ(std::get<0>(results.published()[0]), std::string("(keep)"));
    ASSERT_EQ(std::get<0>(results.published()[1]), std::string("(merge)"));
    ASSERT_EQ(std::get<0>(results.published()[2]), std::string("(tier)"));
    ASSERT_TRUE(&results.retractions() == &fixture.retractedVacuousTheorems);
}

TEST(prover_interface, candidate_registration_preserves_input_multiplicity) {
    gl::tests::ProverFixture fixture("Peano");
    gl::Prover& prover = fixture;
    auto& construction = prover.orConstruction();
    ASSERT_EQ(construction.candidateCount(), size_t(0));
    const gl::OrCandidate candidate{"(existence)", "(companion)", false, false};
    construction.addCandidate(candidate);
    construction.addCandidate(candidate);
    ASSERT_EQ(construction.candidateCount(), size_t(2));
    ASSERT_EQ(fixture.orCandidates.front().existenceTheorem, candidate.existenceTheorem);
    ASSERT_EQ(fixture.orCandidates.front().companionTheorem, candidate.companionTheorem);
}
