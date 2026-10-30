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
#include "../memory.hpp"
#include <cstddef>
#include <utility>
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "prover_initialization_fixture.hpp"
#include "compiled_definition_fixture.hpp"
#include "../prover.hpp"
#include <sstream>
#include <iostream>

TEST(prover_initialization, construction_configures_the_existing_batch_state) {
    gl::tests::ProverFixture engine("Peano", gl::Phase2Backend::cpu);
    ASSERT_EQ(engine.anchorID_, std::string("Peano"));
    ASSERT_TRUE(engine.phase2Backend == gl::Phase2Backend::cpu);
    ASSERT_FALSE(engine.definitions.configurations().empty());
    ASSERT_FALSE(engine.definitions.entries().empty());
    ASSERT_EQ(engine.permanentBodies.size(), size_t(1));
    ASSERT_TRUE(engine.permanentBodies.front() == &engine.body);
    ASSERT_GE(engine.logicalCores, 1u);
    [[maybe_unused]] gl::ProverInitialization inputs{engine.parameters, engine.phase2Backend, engine.body, engine.allMappingsAna, engine.allPermutationsAna, engine.globalDependencies, engine.definitions, engine.anchorInfo, engine.anchorID_, engine.permanentBodies, engine.logicalCores, engine.structuralCompiler, engine.supportingRuleCompiler, engine.compilationPersistence, engine.compilationReuse};
    ASSERT_EQ(engine.permanentBodies.size(), size_t(1));
}

TEST(prover_initialization, enumerates_mapping_ranges_without_reordering) {
    const auto table = gl::tests::ProverInitializationFixture::mappings(3, 3);
    ASSERT_EQ(table.size(), size_t(4));
    const int64_t key = (int64_t(2) << 32) | 2;
    const std::vector<std::vector<int>> expected = {{0,0}, {0,1}, {1,0}, {1,1}};
    ASSERT_TRUE(table.at(key) == expected);
    ASSERT_TRUE(gl::tests::ProverInitializationFixture::mappings(1, 3).empty());
}

TEST(prover_initialization, prepares_an_empty_grid_with_the_original_mail_policy) {
    gl::tests::ProverFixture engine("Peano");
    std::vector<std::string> conjectures;
    std::unordered_map<std::string, std::string> mapping;
    engine.proofGridBuilder.prepareBatch(conjectures, mapping,
        engine.structuralCompiler, engine.orTheoremBuilder, engine.globalDependencies,
        engine.blockMailProcessor, engine.dormantLogicBlocksAtGridBuild,
        engine.rollingMailHistoryEnabled);
    ASSERT_TRUE(conjectures.empty());
    ASSERT_TRUE(mapping.empty());
    ASSERT_EQ(engine.permanentBodies.size(), size_t(1));
    ASSERT_EQ(engine.dormantLogicBlocksAtGridBuild, 0);
    ASSERT_TRUE(engine.rollingMailHistoryEnabled);
}

TEST(prover_initialization, reports_completion_without_changing_proof_results) {
    gl::tests::ProverFixture engine("Peano");
    gl::ProofRunReporter reporter(engine.permanentBodies, engine.globalTheoremList);
    std::ostringstream output;
    struct Capture {
        std::streambuf* previous;
        ~Capture() { std::cout.rdbuf(previous); }
    } capture{std::cout.rdbuf(output.rdbuf())};
    reporter.reportCompletion(engine.permanentBodies.size(), ((engine.body).startInt),
        std::chrono::high_resolution_clock::now());
    const auto bytes = output.str();
    ASSERT_TRUE(bytes.find("Prover finished.") != std::string::npos);
    ASSERT_TRUE(bytes.find("Number proven theorems: 0") != std::string::npos);
    ASSERT_TRUE(bytes.find("permanent_mints_during_prover=0") != std::string::npos);
    ASSERT_TRUE(engine.globalTheoremList.empty());
}

namespace {

// Every field of two compiled entities agrees (LogicalEntity::operator==
// compares only the identity fields).
bool sameEntity(const gl::LogicalEntity& a, const gl::LogicalEntity& b) {
    if (!(a == b) || a.symmetricDigitSlots != b.symmetricDigitSlots
        || a.implications != b.implications
        || a.integrationImplications != b.integrationImplications
        || a.contrapositionPrepared != b.contrapositionPrepared
        || a.contrapositionAuxiliary != b.contrapositionAuxiliary
        || a.conjunctionRefutations != b.conjunctionRefutations
        || a.conjunctionRefutationsPrepared != b.conjunctionRefutationsPrepared
        || a.supportingImplications != b.supportingImplications
        || a.composedSupportingImplications != b.composedSupportingImplications
        || a.supportingImplicationsPrepared != b.supportingImplicationsPrepared
        || a.supportingImplicationAuxiliary != b.supportingImplicationAuxiliary
        || a.contrapositives.size() != b.contrapositives.size())
        return false;
    for (std::size_t i = 0; i < a.contrapositives.size(); ++i) {
        if (a.contrapositives[i].sources != b.contrapositives[i].sources
            || a.contrapositives[i].sourceRule != b.contrapositives[i].sourceRule
            || a.contrapositives[i].reverseRule != b.contrapositives[i].reverseRule)
            return false;
    }
    return true;
}

// Every member of two definition registries agrees.
bool sameRegistry(gl::CompiledDefinitionRegistry& a, gl::CompiledDefinitionRegistry& b) {
    using F = gl::CompiledDefinitionFixture;
    const auto& ea = F::compiledExpressions(a);
    const auto& eb = F::compiledExpressions(b);
    if (ea.size() != eb.size()) return false;
    for (auto ia = ea.begin(), ib = eb.begin(); ia != ea.end(); ++ia, ++ib)
        if (ia->first != ib->first || !sameEntity(ia->second, ib->second)) return false;
    const auto& ca = F::coreExpressionMap(a);
    const auto& cb = F::coreExpressionMap(b);
    if (ca.size() != cb.size()) return false;
    for (auto ia = ca.begin(), ib = cb.begin(); ia != ca.end(); ++ia, ++ib) {
        if (ia->first != ib->first || ia->second.arity != ib->second.arity
            || ia->second.definition != ib->second.definition
            || ia->second.signature != ib->second.signature
            || ia->second.definitionSets != ib->second.definitionSets
            || ia->second.inputArgs != ib->second.inputArgs
            || ia->second.outputArgs != ib->second.outputArgs
            || ia->second.inputIndices != ib->second.inputIndices
            || ia->second.outputIndices != ib->second.outputIndices)
            return false;
    }
    const auto& ta = F::typedEqualityEntities(a);
    const auto& tb = F::typedEqualityEntities(b);
    if (ta.size() != tb.size()) return false;
    for (std::size_t i = 0; i < ta.size(); ++i)
        if (ta[i].label != tb[i].label || !sameEntity(ta[i].entity, tb[i].entity)) return false;
    return F::operators(a) == F::operators(b)
        && F::expressionsFromConfig(a) == F::expressionsFromConfig(b)
        && F::baseExpressionNames(a) == F::baseExpressionNames(b)
        && F::implCounter(a) == F::implCounter(b)
        && F::existenceCounter(a) == F::existenceCounter(b)
        && F::andCounter(a) == F::andCounter(b)
        && F::orCounter(a) == F::orCounter(b)
        && F::variableCounter(a) == F::variableCounter(b);
}

// Every member of two reuse states agrees.
bool sameReuse(const gl::CompilationReuseState& a, const gl::CompilationReuseState& b) {
    if (a.compilationReplacements.size() != b.compilationReplacements.size()) return false;
    for (auto ia = a.compilationReplacements.begin(), ib = b.compilationReplacements.begin();
         ia != a.compilationReplacements.end(); ++ia, ++ib) {
        if (ia->first != ib->first || ia->second.target != ib->second.target
            || ia->second.arguments != ib->second.arguments)
            return false;
    }
    return a.repetitionExclusionMap == b.repetitionExclusionMap
        && a.namedRepetitionExclusionMap == b.namedRepetitionExclusionMap
        && a.compiledOrReuse == b.compiledOrReuse
        && a.namedCompilationOrder == b.namedCompilationOrder
        && a.namedCompilationNames == b.namedCompilationNames
        && a.compilationNamesFrozen == b.compilationNamesFrozen;
}

}  // namespace

TEST(prover_initialization, precompiled_definitions_reproduce_a_fresh_compile) {
    // A prover that installs the fixture's recorded compile holds exactly the
    // state of an independent fresh compile of the same configuration: the
    // compile is deterministic and the registry plus the reuse state are its
    // complete output. FTA carries the integration families and
    // contraposition templates; Peano is the configuration most tests use.
    for (const char* anchor : { "Peano", "FTA" }) {
        gl::tests::ProverFixture recorder(anchor);
        const gl::PrecompiledDefinitions* record =
            gl::tests::precompiledDefinitionsFor(anchor, std::nullopt);
        ASSERT_TRUE(record != nullptr);
        gl::tests::ProverFixture installed(*record, anchor);
        gl::tests::ProverFixture fresh(gl::tests::ProverFixture::FreshCompile{}, anchor);
        ASSERT_FALSE(gl::CompiledDefinitionFixture::compiledExpressions(fresh.definitions).empty());
        ASSERT_FALSE(fresh.compilationReuse.state().repetitionExclusionMap.empty());
        ASSERT_TRUE(sameRegistry(installed.definitions, fresh.definitions));
        ASSERT_TRUE(sameReuse(installed.compilationReuse.state(), fresh.compilationReuse.state()));
        ASSERT_TRUE(sameRegistry(recorder.definitions, fresh.definitions));
        // Everything outside the compile is built by the prover itself.
        ASSERT_EQ(installed.anchorID_, std::string(anchor));
        ASSERT_TRUE(installed.allMappingsAna == fresh.allMappingsAna);
        ASSERT_TRUE(installed.allPermutationsAna == fresh.allPermutationsAna);
        ASSERT_EQ(installed.permanentBodies.size(), std::size_t(1));
        ASSERT_TRUE(installed.permanentBodies.front() == &installed.body);
    }
}

TEST(prover_initialization, fixture_records_the_first_compile_and_installs_it_later) {
    // The fixture keys its record by anchor: a later fixture of the anchor
    // installs it; a fixture with a compiled-registry input never uses it.
    gl::tests::ProverFixture engine("Peano");
    const gl::PrecompiledDefinitions* record =
        gl::tests::precompiledDefinitionsFor("Peano", std::nullopt);
    ASSERT_TRUE(record != nullptr);
    ASSERT_TRUE(gl::tests::precompiledDefinitionsFor("Peano", std::filesystem::path("unused.json")) == nullptr);
    ASSERT_TRUE(gl::tests::precompiledDefinitionsFor("NoSuchAnchor", std::nullopt) == nullptr);
    gl::tests::ProverFixture later("Peano");
    gl::CompiledDefinitionRegistry recordedRegistry = record->registry;
    ASSERT_TRUE(sameRegistry(later.definitions, recordedRegistry));
    ASSERT_TRUE(sameReuse(later.compilationReuse.state(), record->reuse));
}
