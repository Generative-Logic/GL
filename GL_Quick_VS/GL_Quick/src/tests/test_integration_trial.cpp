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
/// @brief Unit tests for integration trial scopes
///        (D-481).
///
/// @details
/// A gated integration admission is made first in its own trial scope
/// `<parent>_boundary_integration_<names>`, named by the witnesses it mints,
/// where they only fire and join equality classes; once the gate elements of
/// every new name are known there, the parent admits the recorded witnesses
/// under the same names with the recorded expressions, and the trial scope is
/// wiped. Covers the scope classifier, the trial-record lifecycle across scope
/// wipes, the gate predicate, trial opening, the restrictions inside a trial
/// and promotion. The disintegration-level cases of the witness-link gate
/// (D-529) share this fixture; its helper-level cases
/// are in `test_witness_link_gate.cpp`.

#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "compiled_definition_fixture.hpp"
#include "integration_processor_fixture.hpp"
#include "../logical_block/scope_syntax.hpp"
#include "../prover.hpp"
#include "../compiler.hpp"

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {
    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief An open trial record at @p trialScope with one gate.
    gl::IntegrationTrialRec openTrial(gl::NameId trialScope, gl::NameId gate) {
        gl::IntegrationTrialRec trial;
        trial.trialScope = trialScope;
        trial.source = 40;
        trial.groupCount = 1;
        trial.groupEnd[0] = 1;
        trial.gates[0] = gate;
        return trial;
    }

    /// @brief A block with the one-witness existence `existence96`, whose
    ///        witness `w` has an operator fact `in3[u_2,u_3,w,u_4]`, and
    ///        helpers for demands, parks and wakes on its marker template.
    struct TrialFixture {
        gl::tests::ProverFixture analyzer{"Peano"};
        gl::Memory memory;
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        std::string source = "(existence96[1,2,3,4])";
        std::string marker = "(in3[2,3,marker,4])";
        int levels[2] = {0, 2};

        TrialFixture() {
            gl::CompiledDefinitionFixture::operators(analyzer.definitions).clear();
            gl::CompiledDefinitionFixture::operators(analyzer.definitions).insert("in3");
            gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence96"] =
                gl::LogicalEntity("existence", {"(in[w,u_1])", "(in3[u_2,u_3,w,u_4])"},
                    "(existence96[u_1,u_2,u_3,u_4])", 4);
            analyzer.supportingRuleCompiler.compileDisintegrationPackages();
            analyzer.parameters.trackHistory = true;
            memory.level = 2;
            memory.nameMap.encode(sp("main"));
        }

        /// @brief Make the source `existence97[1,2,3,4]`: a witness `v` with
        ///        `in[v,u_1]` and the nested existence `existence98`, whose
        ///        witness `w` has the route-3 constituent `in3[u_2,u_3,w,u_4]`
        ///        and the fact `in2[v,w,u_2]` relating it to `v`.
        void defineNestedSource() {
            gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence97"] =
                gl::LogicalEntity("existence", {"(in[v,u_1])", "(existence98[v,u_2,u_3,u_4])"},
                    "(existence97[u_1,u_2,u_3,u_4])", 4);
            gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence98"] =
                gl::LogicalEntity("existence", {"(in3[u_2,u_3,w,u_4])", "(in2[u_1,w,u_2])"},
                    "(existence98[u_1,u_2,u_3,u_4])", 4);
            analyzer.supportingRuleCompiler.compileDisintegrationPackages();
            source = "(existence97[1,2,3,4])";
        }

        /// @brief Make the source `existence93[1,2,3,4,5]`: a witness `w`
        ///        with the route-3 constituent `in3[u_2,u_3,w,u_4]` and the
        ///        other part `in2[u_1,w,u_5]`, which links `w` to `u_1` (a
        ///        number like `w`; `u_5` is a function) for the witness-link
        ///        gate (D-529).
        void defineLinkedSource() {
            gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions)["existence93"] =
                gl::LogicalEntity("existence", {"(in3[u_2,u_3,w,u_4])", "(in2[u_1,w,u_5])"},
                    "(existence93[u_1,u_2,u_3,u_4,u_5])", 5);
            analyzer.supportingRuleCompiler.compileDisintegrationPackages();
            source = "(existence93[1,2,3,4,5])";
        }

        /// @brief Mint the scope `<parent>_boundary_<payload>` and initialize
        ///        its canonical state; returns its name.
        std::string scope(gl::NameId parent, const char* payload) {
            const gl::NameId id = memory.nameMap.encodePush(parent, sp(payload));
            analyzer.canonicalizer.initializeCanonicalScope(memory, id);
            return memory.nameMap.decode(id);
        }

        /// @brief An integration demand on the marker template at @p at.
        void demand(const std::string& at) {
            memory.overallHashMemory.admissionSetIntegration.mint(
                gl::mintTemplateKey(memory.templateInterner, memory.nameMap, marker, at));
        }

        /// @brief Disintegrate the source at @p at; returns whether every
        ///        construction group is covered.
        bool disintegrate(const std::string& at, bool forced, gl::DisintProducts& output) {
            return analyzer.disintegrationProcessor.disintegrateExpr2(gl::StrSpan(source), memory, 0,
                forced, gl::StrSpan(at), output, true, false, levels, 2, true);
        }

        /// @brief Park the source's integration construction at @p at
        ///        (`main` by default); returns the park key.
        int64_t park(const std::string& at = std::string("main")) {
            analyzer.rejectedSources.updateRejectedMapIntegration(gl::StrSpan(marker), gl::StrSpan(source),
                0, 2, levels, 2, memory.overallHashMemory, memory, gl::StrSpan(at), nullptr, 0, 0);
            return gl::mintTemplateKey(memory.templateInterner, memory.nameMap, marker, at);
        }

        /// @brief Wake the parked integration constructions for a demand at
        ///        @p at.
        void wake(const std::string& at) {
            analyzer.rejectedSourceReplay.revisitRejectedIntegration2(gl::StrSpan(marker), memory,
                gl::StrSpan(at));
        }

        /// @brief The trial record key of the source at `main`.
        int64_t trialKey() {
            return gl::packOriginKey(analyzer.rejectedSources.canonicalExistenceId(memory,
                gl::StrSpan(source), gl::NameMap::MAIN_ID), gl::NameMap::MAIN_ID);
        }

        /// @brief The source arrives at @p at (`main` by default) through the
        ///        deposit door as a local statement, so it is known there with
        ///        its levels and history.
        void depositLocal(const std::string& at = std::string("main")) {
            analyzer.statementMemory.addExprToMemoryBlock(gl::StrSpan(source), memory, 0, 1, levels, 2,
                gl::TransientOrigin{true, gl::OriginTag::taskFormulation, nullptr, 0},
                -1, -1, gl::StrSpan(at), false);
        }

        /// @brief The receipt state of the source's construction at @p at.
        int32_t receipt(const std::string& at) {
            const int32_t* state = memory.existenceReplays.find(gl::packOriginKey(
                memory.nameMap.encode(gl::StrSpan(source)), memory.nameMap.encode(gl::StrSpan(at))));
            return state == nullptr ? 0 : *state;
        }

        /// @brief Run one ordinary absorb of the internal mail.
        void absorb() {
            analyzer.logicBlockOperations.standardProcessing(memory, nullptr,
                memory.sameIterationInternalMail, memory.nextIterationInternalMail, -1);
        }
    };

    /// @brief One entity of a marked integration instruction (tests only).
    struct EntitySpec {
        const char* category;
        std::string signature;
        int32_t arity;
        std::vector<std::string> elements;
    };

    /// @brief File the marked integration instruction made of @p entities
    ///        under @p key at @p scope (`main` by default), with the path
    ///        list @p pathList (empty for none, D-529) and the marked goal
    ///        @p markedGoal.
    void fileEntities(gl::Memory& m, const std::string& key, const std::vector<EntitySpec>& entities,
                      const std::string& pathList = std::string(), const std::string& scope = std::string("main"),
                      const std::string& markedGoal = std::string("(goal54[1,2,3,4,5,6])")) {
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        gl::ScratchScope frame(arena);
        gl::WorkInstruction wi(&arena);
        for (const EntitySpec& entity : entities) {
            const int32_t mark = wi.elemMark();
            for (const std::string& element : entity.elements) wi.addElement(gl::StrSpan(element));
            wi.commitEntity(sp(entity.category), gl::StrSpan(entity.signature), sp(""), entity.arity, mark);
        }
        // By default the root goal is distinct from the existence entity, as
        // in a real instruction (fold over its inner existence).
        wi.setMarkedGoal(gl::StrSpan(markedGoal));
        wi.setPathVariables(gl::StrSpan(pathList));
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, scope);
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        entries.findOrEmplaceWork(wi, m.valueInterner);
        entries.writeToCold(m.overallHashMemory.admissionMapIntegration, pk);
    }

    /// @brief The entities of the instruction `fileInstruction` files: an
    ///        existence listing @p key with the and side `and56`, whose
    ///        elements are @p andElements; a nonempty @p nestedElements adds
    ///        the and `and57[u_5,marker]` with those elements.
    std::vector<EntitySpec> instructionEntities(const std::string& key,
                                                const std::vector<std::string>& andElements,
                                                const std::vector<std::string>& nestedElements) {
        std::vector<EntitySpec> entities = {
            { "existence", "(existence55[u_2,u_3,u_4,u_5,u_6])", 5, { key, "(and56[u_5,u_6,marker])" } },
            { "and", "(and56[u_5,u_6,marker])", 3, andElements } };
        if (!nestedElements.empty()) entities.push_back({ "and", "(and57[u_5,marker])", 2, nestedElements });
        return entities;
    }

    /// @brief File a marked integration instruction under @p key at `main`:
    ///        the entities of `instructionEntities`.
    void fileInstruction(gl::Memory& m, const std::string& key,
                         const std::vector<std::string>& andElements,
                         const std::vector<std::string>& nestedElements = {}) {
        fileEntities(m, key, instructionEntities(key, andElements, nestedElements));
    }

    /// @brief The entities of an instruction whose key existence has the
    ///        existence `existence59` beside @p key; its bound variable
    ///        @p bound is filled through `in2[<bound>,marker,u_2]` and its and
    ///        side `and60` carries @p gateElements.
    std::vector<EntitySpec> coadmissionEntities(const std::string& key,
                                                const std::vector<std::string>& gateElements,
                                                const std::string& bound) {
        return {
            { "existence", "(existence55[u_2,u_3,u_4,u_5,u_6])", 5, { key, "(existence59[u_2,u_5,u_6,marker])" } },
            { "existence", "(existence59[u_2,u_5,u_6,marker])", 4,
                { "(in2[" + bound + ",marker,u_2])", "(and60[u_5,u_6," + bound + "])" } },
            { "and", "(and60[u_5,u_6," + bound + "])", 3, gateElements } };
    }

    /// @brief File, under @p key at `main`, the instruction of
    ///        `coadmissionEntities` with the bound variable `pi_1`.
    void fileCoadmissionInstruction(gl::Memory& m, const std::string& key,
                                    const std::vector<std::string>& gateElements) {
        fileEntities(m, key, coadmissionEntities(key, gateElements, "pi_1"));
    }

    /// @brief Register the entities of a hand-built instruction as registry
    ///        entities carrying their compiled integration sets.
    ///
    /// @details
    /// Production instructions are instantiated from the registry, and Case B
    /// of `prepareIntegrationCore2` takes a table entity's integration set
    /// from the registry entity of its name
    /// (D-503). A test whose admission commits
    /// a hand-built instruction registers it first: the arguments of each
    /// table signature become the tokens `u_1`.. in order, and the one other
    /// argument of an existence's elements becomes its witness `1`. An
    /// implication an element names that the registry lacks is registered as
    /// `in[u_1,u_2]` implies `in[u_2,u_1]`; one the registry holds must have
    /// the element's arity. The counters move past the
    /// hand-picked numbers, so the compacts minted next take fresh names; the
    /// sets are compiled by the production compiler once every entity is in
    /// place.
    void registerTableEntities(gl::tests::ProverFixture& analyzer, const std::vector<EntitySpec>& entities) {
        auto& compiled = gl::CompiledDefinitionFixture::compiledExpressions(analyzer.definitions);
        std::vector<std::string> names;
        for (const EntitySpec& entity : entities) {
            const std::string category = entity.category;
            assert((category == "and" || category == "existence")
                && "registerTableEntities: an instruction lists and and existence entities");
            const std::string tableSignature = entity.signature;
            const std::string name = tableSignature.substr(1, tableSignature.find('[') - 1);
            const std::vector<std::string> tableArgs = ce::getArgs(tableSignature);
            assert(static_cast<int32_t>(tableArgs.size()) == entity.arity);
            std::map<std::string, std::string> tokens;
            std::string signature = "(" + name + "[";
            for (std::size_t i = 0; i < tableArgs.size(); ++i) {
                tokens[tableArgs[i]] = "u_" + std::to_string(i + 1);
                signature += (i == 0 ? "" : ",") + tokens[tableArgs[i]];
            }
            signature += "])";
            int32_t witnesses = 0;
            for (const std::string& element : entity.elements) {
                for (const std::string& argument : ce::getArgs(element)) {
                    if (tokens.count(argument) != 0) continue;
                    tokens[argument] = "1";
                    ++witnesses;
                }
            }
            assert(witnesses == (category == "existence" ? 1 : 0)
                && "registerTableEntities: an existence binds one witness, an and none");
            std::vector<std::string> elements;
            for (const std::string& element : entity.elements) {
                elements.push_back(ce::replaceKeysInString(element, tokens));
                const std::string core = element.substr(1, element.find('[') - 1);
                if (core.rfind("implication", 0) != 0) continue;
                const auto known = compiled.find(core);
                if (known != compiled.end()) {
                    assert(ce::getArgs(known->second.signature).size() == ce::getArgs(element).size()
                        && "registerTableEntities: an element names a registry implication of another arity");
                    continue;
                }
                assert(ce::getArgs(element).size() == 2 && "registerTableEntities: a binary implication");
                compiled[core] = gl::LogicalEntity("implication", { "(in[u_1,u_2])", "(in[u_2,u_1])" },
                    "(" + core + "[u_1,u_2])", 2);
            }
            compiled[name] = gl::LogicalEntity(category, elements, signature, entity.arity);
            names.push_back(name);
        }
        int* counters[] = { &gl::CompiledDefinitionFixture::implCounter(analyzer.definitions),
            &gl::CompiledDefinitionFixture::existenceCounter(analyzer.definitions),
            &gl::CompiledDefinitionFixture::andCounter(analyzer.definitions),
            &gl::CompiledDefinitionFixture::orCounter(analyzer.definitions) };
        for (int* counter : counters) *counter = std::max(*counter, 1000);
        for (const std::string& name : names) analyzer.integrationRuleCompiler.compileIntegrationImplications(name);
    }

    /// @brief File the instruction of `instructionEntities` under @p key at
    ///        `main` with its entities registered (`registerTableEntities`),
    ///        for a test whose admission commits it; @p pathList as in
    ///        `fileEntities`.
    void fileRegisteredInstruction(gl::tests::ProverFixture& analyzer, gl::Memory& m, const std::string& key,
                                   const std::vector<std::string>& andElements,
                                   const std::string& pathList = std::string()) {
        const std::vector<EntitySpec> entities = instructionEntities(key, andElements, {});
        registerTableEntities(analyzer, entities);
        fileEntities(m, key, entities, pathList);
    }

    /// @brief The values recorded under the first instruction of @p key at
    ///        `main` (the witnesses the instruction was committed for).
    int32_t committedValues(gl::Memory& m, const std::string& key) {
        gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
        gl::ScratchScope frame(arena);
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, std::string("main"));
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        assert(entries.entries.size() > 0);
        return entries.entries[0].valCount;
    }

    /// The route-3 key of TrialFixture's witness constituent `in3[2,3,w,4]`.
    const std::string kTrialKey = "(in3[u_2,u_3,marker,u_4])";

    /// @brief The witnesses @p trial lists, in slot order; their slot
    ///        ordinals below the trial slot go to @p ordinals when given.
    std::vector<std::string> trialWitnesses(gl::Memory& m, const gl::IntegrationTrialRec& trial,
                                            std::vector<int32_t>* ordinals = nullptr) {
        std::vector<std::string> names;
        for (int32_t i = 0; i < trial.witnessCount; ++i) {
            const int64_t* entry = m.lbMemory.integrationTrialContents.find(gl::packInt32Pair(trial.trialScope, i));
            assert(entry != nullptr);
            if (ordinals != nullptr) ordinals->push_back(static_cast<int32_t>(static_cast<uint64_t>(*entry) >> 32));
            names.push_back(m.nameMap.decode(static_cast<gl::NameId>(static_cast<uint32_t>(*entry))));
        }
        return names;
    }

    /// @brief The statements @p trial lists for its promotion.
    std::vector<std::string> trialStatements(gl::Memory& m, const gl::IntegrationTrialRec& trial) {
        std::vector<std::string> statements;
        for (int32_t i = trial.witnessCount; i < trial.witnessCount + trial.expressionCount; ++i) {
            const int64_t* entry = m.lbMemory.integrationTrialContents.find(gl::packInt32Pair(trial.trialScope, i));
            assert(entry != nullptr);
            if (static_cast<int32_t>(static_cast<uint64_t>(*entry) >> 32) != gl::kIntegrationTrialStatement) continue;
            statements.push_back(m.nameMap.decode(static_cast<gl::NameId>(static_cast<uint32_t>(*entry))));
        }
        return statements;
    }

    /// @brief Whether @p output holds @p text at @p validity.
    bool published(gl::DisintProducts& output, const std::string& text, const std::string& validity) {
        bool found = false;
        output.statements.forEachSorted([&](gl::StrSpan s, gl::StrSpan v) {
            if (gl::equalSpans(s, gl::StrSpan(text)) && gl::equalSpans(v, gl::StrSpan(validity))) found = true;
        });
        return found;
    }

    /// @brief Make the gates of the source's trial's first group known at
    ///        its trial scope and drain: the trial passes and the deposited
    ///        source is replayed at main.
    void passTrial(TrialFixture& f) {
        const gl::IntegrationTrialRec* trial = f.memory.lbMemory.integrationTrials.find(f.trialKey());
        assert(trial != nullptr && trial->groupCount > 0);
        const gl::NameId trialScope = trial->trialScope;
        for (int32_t i = 0; i < trial->groupEnd[0]; ++i)
            gl::upsertStatementKey(f.memory.intKnownStatements,
                gl::packStatementKey(trial->gates[i], trialScope), true);
        f.analyzer.implicationSubproofs.drainIntegrationTrials(f.memory);
        assert(f.memory.lbMemory.integrationTrials.find(f.trialKey())->status == gl::IntegrationTrialRec::kPassed);
    }
} // namespace

// The payload test is a prefix test: a trial payload names a witness after
// `integration_`; the hypothetical-disintegration sentinel, whose payload
// contains `integration_` inside, and every other payload kind are not trials.
TEST(integration_trial, payload_prefix_classifies_trial_scopes) {
    using gl::ScopeSyntax;
    ASSERT_TRUE(ScopeSyntax::isIntegrationTrialPayload(sp("integration_int_0_lev_6_119")));
    ASSERT_TRUE(ScopeSyntax::isIntegrationTrialPayload(sp("integration_int_0_lev_6_119_int_0_lev_6_120")));
    ASSERT_FALSE(ScopeSyntax::isIntegrationTrialPayload(sp("integration_")));
    ASSERT_FALSE(ScopeSyntax::isIntegrationTrialPayload(
        sp("product_of_hypo_disintegration_of_integration_goal_(in[a,b])")));
    ASSERT_FALSE(ScopeSyntax::isIntegrationTrialPayload(sp("orint_(or0[a,b])_((in[a,b]))")));
    ASSERT_FALSE(ScopeSyntax::isIntegrationTrialPayload(sp("(implication7[a])")));
    ASSERT_FALSE(ScopeSyntax::isIntegrationTrialPayload(sp("")));
}

// A trial scope and every scope below it are detected through the parent
// chain; `main` (also before any name is encoded), sibling scopes and the
// sentinel scope are not.
TEST(integration_trial, trial_scope_and_descendants_are_detected) {
    using gl::ScopeSyntax;
    gl::Memory m;
    ASSERT_FALSE(ScopeSyntax::atOrBelowIntegrationTrial(m.nameMap, gl::NameMap::MAIN_ID));
    const gl::NameId mainId = m.nameMap.encode(sp("main"));
    ASSERT_EQ(mainId, gl::NameMap::MAIN_ID);
    const gl::NameId trial = m.nameMap.encodePush(mainId, sp("integration_int_0_lev_6_119"));
    const gl::NameId inner = m.nameMap.encodePush(trial, sp("(implication7[a])"));
    const gl::NameId sibling = m.nameMap.encodePush(mainId, sp("(implication8[a])"));
    const gl::NameId sentinel = m.nameMap.encodePush(mainId,
        sp("product_of_hypo_disintegration_of_integration_goal_(in[a,b])"));
    ASSERT_EQ(m.nameMap.decode(trial), std::string("main_boundary_integration_int_0_lev_6_119"));
    ASSERT_TRUE(ScopeSyntax::isIntegrationTrialScope(m.nameMap, trial));
    ASSERT_FALSE(ScopeSyntax::isIntegrationTrialScope(m.nameMap, inner));
    ASSERT_FALSE(ScopeSyntax::isIntegrationTrialScope(m.nameMap, mainId));
    ASSERT_TRUE(ScopeSyntax::atOrBelowIntegrationTrial(m.nameMap, trial));
    ASSERT_TRUE(ScopeSyntax::atOrBelowIntegrationTrial(m.nameMap, inner));
    ASSERT_FALSE(ScopeSyntax::atOrBelowIntegrationTrial(m.nameMap, mainId));
    ASSERT_FALSE(ScopeSyntax::atOrBelowIntegrationTrial(m.nameMap, sibling));
    ASSERT_FALSE(ScopeSyntax::atOrBelowIntegrationTrial(m.nameMap, sentinel));
}

// A scope wipe erases an open trial with its trial scope and every trial with
// its parent scope; a passed trial outlives its wiped trial scope as the
// parent's receipt, and a trial under an untouched parent survives. A trial's
// listed witnesses and expressions leave with its record.
TEST(integration_trial, scope_wipe_erases_open_trials_and_keeps_passed_receipts) {
    gl::Memory m;
    const gl::NameId mainId = m.nameMap.encode(sp("main"));
    const gl::NameId parent = m.nameMap.encodePush(mainId, sp("(implication7[a])"));
    const gl::NameId openScope = m.nameMap.encodePush(parent, sp("integration_w1"));
    const gl::NameId passedScope = m.nameMap.encodePush(parent, sp("integration_w2"));
    const gl::NameId mainTrial = m.nameMap.encodePush(mainId, sp("integration_w3"));
    const gl::NameId gate = m.nameMap.encode(sp("(in2[a,b,w1])"));
    const int64_t openKey = gl::packOriginKey(m.nameMap.encode(sp("(existence1[a])")), parent);
    const int64_t passedKey = gl::packOriginKey(m.nameMap.encode(sp("(existence2[a])")), parent);
    const int64_t mainKey = gl::packOriginKey(m.nameMap.encode(sp("(existence3[a])")), mainId);
    gl::IntegrationTrialRec passed = openTrial(passedScope, gate);
    passed.status = gl::IntegrationTrialRec::kPassed;
    m.lbMemory.integrationTrials.insert(openKey, openTrial(openScope, gate));
    m.lbMemory.integrationTrials.insert(passedKey, passed);
    m.lbMemory.integrationTrials.insert(mainKey, openTrial(mainTrial, gate));
    for (const gl::NameId scope : { openScope, passedScope, mainTrial }) {
        m.lbMemory.integrationTrialContents.insert(gl::packInt32Pair(scope, 0), gl::packInt32Pair(0, gate));
        m.lbMemory.integrationTrialContents.insert(gl::packInt32Pair(scope, 1),
            gl::packInt32Pair(gl::kIntegrationTrialStatement, gate));
    }

    const gl::NameId wiped[2] = { openScope, passedScope };
    m.wipeSubtrees(wiped, 2);
    ASSERT_TRUE(m.lbMemory.integrationTrials.find(openKey) == nullptr);
    ASSERT_TRUE(m.lbMemory.integrationTrials.find(passedKey) != nullptr);
    ASSERT_TRUE(m.lbMemory.integrationTrials.find(mainKey) != nullptr);
    ASSERT_TRUE(m.lbMemory.integrationTrialContents.find(gl::packInt32Pair(openScope, 0)) == nullptr);
    ASSERT_TRUE(m.lbMemory.integrationTrialContents.find(gl::packInt32Pair(passedScope, 1)) != nullptr);
    ASSERT_EQ(m.lbMemory.integrationTrialContents.count(), 4);

    m.wipeSubtrees(&parent, 1);
    ASSERT_TRUE(m.lbMemory.integrationTrials.find(passedKey) == nullptr);
    ASSERT_TRUE(m.lbMemory.integrationTrials.find(mainKey) != nullptr);
    ASSERT_EQ(m.lbMemory.integrationTrials.count(), 1);
    ASSERT_TRUE(m.lbMemory.integrationTrialContents.find(gl::packInt32Pair(passedScope, 0)) == nullptr);
    ASSERT_TRUE(m.lbMemory.integrationTrialContents.find(gl::packInt32Pair(mainTrial, 1)) != nullptr);
    ASSERT_EQ(m.lbMemory.integrationTrialContents.count(), 2);
}

// Inside a trial scope and below it a disintegration mints no witness, even
// under an integration demand or when forced (an induction head fired there),
// and parks nothing; the unforced expansion still counts as complete. The same
// disintegration at a sibling scope mints.
TEST(integration_trial, disintegration_inside_a_trial_mints_no_witness) {
    for (const bool forced : {false, true}) {
        TrialFixture f;
        const std::string trial = f.scope(gl::NameMap::MAIN_ID, "integration_int_0_lev_2_9");
        const std::string inner = f.scope(f.memory.nameMap.lookup(gl::StrSpan(trial)), "(implication7[a])");
        for (const std::string& at : {trial, inner}) {
            f.demand(at);
            gl::DisintProducts output(&f.arena);
            const bool full = f.disintegrate(at, forced, output);
            ASSERT_EQ(f.memory.startInt, 0);
            if (!forced) ASSERT_TRUE(full);
            output.statements.forEachSorted([&](gl::StrSpan text, gl::StrSpan) {
                ASSERT_FALSE(gl::containsSpan(text, gl::StrSpan("__gl_temporary_witness_", 23)));
            });
        }
        ASSERT_TRUE(f.memory.overallHashMemory.rejectedMap.empty());
        ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.empty());
        ASSERT_TRUE(f.memory.existenceReplays.empty());

        const std::string sibling = f.scope(gl::NameMap::MAIN_ID, "child");
        f.demand(sibling);
        gl::DisintProducts output(&f.arena);
        f.disintegrate(sibling, forced, output);
        ASSERT_TRUE(f.memory.startInt > 0);
    }
}

// A trial scope prepares no integration: no task, start snapshot or scope is
// made there. The same preparation at main registers its task.
TEST(integration_trial, trial_scope_prepares_no_integration) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    memory.nameMap.encode(sp("main"));
    const gl::NameId trialId = memory.nameMap.encodePush(gl::NameMap::MAIN_ID, sp("integration_int_0_lev_0_7"));
    const std::string trial = memory.nameMap.decode(trialId);
    const std::string expression = "(in[b,N])";
    const std::string unchangeable = "N";
    const gl::StrSpan run[1] = { gl::StrSpan(unchangeable) };
    const int32_t names = memory.nameMap.stackSize();

    analyzer.integrationProcessor.prepareIntegration(gl::StrSpan(expression), run, 1, memory,
        gl::StrSpan(trial), gl::StrSpan());
    ASSERT_EQ(memory.integrationPreparedMarker.count(), 0);
    ASSERT_EQ(memory.integrationStartIntMap.count(), 0);
    ASSERT_EQ(memory.nameMap.stackSize(), names);

    analyzer.integrationProcessor.prepareIntegration(gl::StrSpan(expression), run, 1, memory,
        sp("main"), gl::StrSpan());
    ASSERT_TRUE(memory.integrationPreparedMarker.count() != 0);
    ASSERT_TRUE(memory.integrationStartIntMap.count() != 0);
}

// A full validity name is recognised as a sentinel scope, or a scope below
// one, by the boundary token plus the sentinel prefix; hypothesis, trial and
// ordinary scopes, and the sentinel's former name, are not.
TEST(integration_trial, hypothetical_sentinel_names_are_recognised) {
    using gl::ScopeSyntax;
    ASSERT_TRUE(ScopeSyntax::namesHypotheticalSentinel(
        sp("main_boundary_product_of_hypo_disintegration_of_integration_goal_17_(in[a,b])")));
    ASSERT_TRUE(ScopeSyntax::namesHypotheticalSentinel(
        sp("main_boundary_product_of_hypo_disintegration_of_integration_goal_17_(in[a,b])_boundary_(implication7[a])")));
    ASSERT_TRUE(ScopeSyntax::namesHypotheticalSentinel(
        sp("main_boundary_(implication32[2,11])_boundary_product_of_hypo_disintegration_of_integration_goal_3_(in2[2,x,11])")));
    ASSERT_FALSE(ScopeSyntax::namesHypotheticalSentinel(sp("main")));
    ASSERT_FALSE(ScopeSyntax::namesHypotheticalSentinel(sp("main_boundary_integration_int_0_lev_6_119")));
    ASSERT_FALSE(ScopeSyntax::namesHypotheticalSentinel(sp("main_boundary__var0_x_hypo_(in[x,1])")));
    ASSERT_FALSE(ScopeSyntax::namesHypotheticalSentinel(sp("main_boundary_hypothetical_disintegration")));
    ASSERT_FALSE(ScopeSyntax::namesHypotheticalSentinel(
        sp("product_of_hypo_disintegration_of_integration_goal_17_(in[a,b])")));
}

// The sentinel scope of a hypothetical disintegration serves only its dry run
// (D-496): preparing an integration at main
// opens one numbered sentinel, leaves it queued for the step's wipe and
// filtered; the wipe removes its class view.
TEST(integration_trial, sentinel_scope_is_wiped_after_its_dry_run) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::Memory memory;
    memory.nameMap.encode(sp("main"));
    const std::string expression = "(in[b,N])";
    const std::string unchangeable = "N";
    const gl::StrSpan run[1] = { gl::StrSpan(unchangeable) };
    analyzer.integrationProcessor.prepareIntegration(gl::StrSpan(expression), run, 1, memory,
        sp("main"), gl::StrSpan());

    static const char prefix[] = "product_of_hypo_disintegration_of_integration_goal_";
    const int32_t prefixLen = static_cast<int32_t>(sizeof(prefix) - 1);
    gl::NameId sentinel = 0;
    int32_t sentinels = 0;
    for (gl::NameId id = 2; id <= memory.nameMap.nameCount(); ++id) {
        if (memory.nameMap.stackEmpty(id) || memory.nameMap.parentOf(id) != gl::NameMap::MAIN_ID) continue;
        const gl::StrSpan payload = memory.nameMap.decodeSubView(memory.nameMap.stackBack(id));
        if (payload.len > prefixLen && std::memcmp(payload.ptr, prefix, static_cast<size_t>(prefixLen)) == 0) {
            sentinel = id;
            ++sentinels;
        }
    }
    ASSERT_EQ(sentinels, 1);
    ASSERT_TRUE(gl::ScopeSyntax::namesHypotheticalSentinel(gl::StrSpan(memory.nameMap.decode(sentinel))));
    ASSERT_TRUE(memory.pendingWipeScopes.contains(sentinel));
    ASSERT_TRUE(memory.intValidityNamesToFilter.lookup(sentinel) != 0);
    ASSERT_TRUE(memory.canonicalEquivalenceClassesMap.lookup(sentinel) != 0);

    memory.wipeSubtree(sentinel);
    ASSERT_TRUE(memory.canonicalEquivalenceClassesMap.lookup(sentinel) == 0);
}

// A construction parked at main never replays into a demanding trial scope;
// the park stays and a demanding sibling scope receives the replay.
TEST(integration_trial, parked_existence_never_replays_into_a_trial) {
    TrialFixture f;
    const std::string trial = f.scope(gl::NameMap::MAIN_ID, "integration_int_0_lev_2_9");
    const std::string sibling = f.scope(gl::NameMap::MAIN_ID, "child");
    const int64_t park = f.park();
    f.demand(trial);
    f.wake(trial);
    ASSERT_TRUE(f.memory.sameIterationInternalMail.statements_.empty());
    ASSERT_TRUE(f.memory.existenceReplays.empty());
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.lookup(park) != 0);

    f.demand(sibling);
    f.wake(sibling);
    ASSERT_FALSE(f.memory.sameIterationInternalMail.statements_.empty());
}

// A negated numbered conjunction deposited as an integration premise opens
// its case split at an ordinary scope; inside a trial scope it is a plain
// fact: no case split opens and nothing parks.
TEST(integration_trial, no_case_split_opens_inside_a_trial) {
    for (const bool trial : {false, true}) {
        gl::tests::ProverFixture ea(std::string("Peano"));
        gl::CompiledDefinitionFixture::compiledExpressions(ea.definitions)["and987"] = gl::LogicalEntity("and",
            {"(in[u_1,u_2])", "!(in[u_2,u_1])"}, "(and987[u_1,u_2])", 2);
        ea.body.primedForContradiction = true;
        ea.body.isActive = true;
        ea.body.nameMap.encode(sp("main"));
        const gl::NameId scopeId = ea.body.nameMap.encodePush(gl::NameMap::MAIN_ID,
            sp(trial ? "integration_int_0_lev_0_7" : "child"));
        const std::string scope = ea.body.nameMap.decode(scopeId);
        const std::string premise = "!(and987[a,N])";
        const int levels[1] = { 0 };
        const gl::TransientOrigin origin{ true, gl::OriginTag::premiseElement, nullptr, 0 };
        ea.statementMemory.addExprToMemoryBlock(gl::StrSpan(premise), ea.body, 0, 0, levels, 1,
            origin, -1, -1, gl::StrSpan(scope),
            /*doNotDisintegrate=*/false, /*allowOrDisintegration=*/false,
            /*registeredOut=*/nullptr, /*recursionHead=*/false,
            /*convergenceMailOut=*/nullptr, /*replayKinds=*/0,
            /*replayIntegrationDepth=*/-1, /*integrationPremise=*/true);

        ASSERT_EQ(ea.body.pendingOrReleases.count(), trial ? 0 : 1);
        ASSERT_EQ(ea.body.orDisjunctCount.empty(), trial);
        ASSERT_TRUE(ea.body.overallHashMemory.rejectedMapOrdis.empty());
        ASSERT_TRUE(ea.body.overallHashMemory.rejectedMapOrdis2.empty());
        ASSERT_TRUE(gl::lookupStatementFlags(ea.body.intKnownStatements, ea.body.nameMap,
            gl::StrSpan(premise), gl::StrSpan(scope)) != nullptr);
    }
}

// The parent's publication pass skips every witness on trial; a trial pass
// publishes its own witness and the slots below it, the names minted with it;
// no filter publishes everything.
TEST(integration_trial, publication_filter_selects_parent_and_trial_passes) {
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    gl::WitnessMetaStore store(arena);
    store.append(gl::TemporaryWitnessKind::Integration, 0, 0, 0, sp("(existence1[a])"));
    store.append(gl::TemporaryWitnessKind::Integration, 0, 1, 1, sp("(existence2[a])"));
    store.closeSubtree(1);
    store.append(gl::TemporaryWitnessKind::Algebra, 0, 0, 2, sp("(existence3[a])"));
    for (int32_t id = 1; id <= 3; ++id) ASSERT_TRUE(store.publishes(id));
    const uint8_t onTrial[4] = { 0, 1, 0, 0 };
    store.setPublicationFilter(onTrial, 0);
    ASSERT_FALSE(store.publishes(1));
    ASSERT_TRUE(store.publishes(3));
    store.setPublicationFilter(onTrial, 1);
    ASSERT_TRUE(store.publishes(1));
    ASSERT_TRUE(store.publishes(2));
    ASSERT_FALSE(store.publishes(3));
    store.setPublicationFilter(nullptr, 0);
    for (int32_t id = 1; id <= 3; ++id) ASSERT_TRUE(store.publishes(id));
}

// A gate element is a non-implication element whose arguments are all fixed
// or the argument the witness fills (the marker on its own route); another
// bound variable, a pattern argument or an implication never gates.
TEST(integration_trial, gate_elements_are_fixed_non_implications) {
    using AM = gl::AdmissionMemory;
    gl::tests::ProverFixture ea(std::string("FTA"));
    const gl::StrSpan marker = sp("marker");
    ASSERT_TRUE(AM::isIntegrationTrialGateElement(sp("(in2[u_10,u_14,marker])"), marker, ea.definitions));
    ASSERT_TRUE(AM::isIntegrationTrialGateElement(sp("!(in2[u_1,marker])"), marker, ea.definitions));
    ASSERT_TRUE(AM::isIntegrationTrialGateElement(sp("(in[u_7,u_1])"), marker, ea.definitions));
    ASSERT_FALSE(AM::isIntegrationTrialGateElement(sp("(implication32[u_1,marker])"), marker, ea.definitions));
    // A configured definition whose whole body is one implication never gates.
    ASSERT_FALSE(AM::isIntegrationTrialGateElement(sp("(subset[u_1,marker])"), marker, ea.definitions));
    ASSERT_TRUE(AM::isIntegrationTrialGateElement(sp("!(subset[u_1,marker])"), marker, ea.definitions));
    ASSERT_FALSE(AM::isIntegrationTrialGateElement(sp("(in2[u_5,x,marker])"), marker, ea.definitions));
    ASSERT_FALSE(AM::isIntegrationTrialGateElement(sp("(in2[u_5,pi_1,marker])"), marker, ea.definitions));
    // On a coadmission route the witness fills a bound variable; the marker
    // then names another witness and never gates.
    ASSERT_TRUE(AM::isIntegrationTrialGateElement(sp("(in2[u_5,u_6,pi_1])"), sp("pi_1"), ea.definitions));
    ASSERT_FALSE(AM::isIntegrationTrialGateElement(sp("(in2[u_5,marker,pi_1])"), sp("pi_1"), ea.definitions));
    ASSERT_FALSE(AM::isIntegrationTrialGateElement(sp("(in2[u_5,u_6,marker])"), sp("pi_1"), ea.definitions));
}

// The gates of an instruction are the fixed non-implication elements of the
// ands on the sides of the existence listing the key, nested ands included:
// a marker gate names the witness, a fixed gate counts while unknown. A
// repeated instruction adds no group; an instruction without an unknown gate
// is ungated, and a constituent without a key matches nothing.
TEST(integration_trial, trial_gates_walk_the_ands_of_the_key_existence) {
    const std::vector<std::string> andSide = { "(and57[u_5,marker])", "(implication58[u_6,marker])",
                                               "(in2[u_5,x,marker])" };
    const std::vector<std::string> nested = { "(in2[u_5,u_7,marker])", "(in[u_7,u_1])" };
    {
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileInstruction(f.memory, kTrialKey, andSide, nested);
        ASSERT_TRUE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, nullptr));
        ASSERT_TRUE(f.memory.nameMap.lookup(sp("(in2[5,7,w1])")) == 0);
        gl::IntegrationTrialRec record;
        ASSERT_TRUE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, &record));
        ASSERT_EQ(record.groupCount, 1);
        ASSERT_EQ(record.groupEnd[0], 2);
        ASSERT_EQ(record.gates[0], f.memory.nameMap.lookup(sp("(in2[5,7,w1])")));
        ASSERT_EQ(record.gates[1], f.memory.nameMap.lookup(sp("(in[7,1])")));
        ASSERT_TRUE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, &record));
        ASSERT_EQ(record.groupCount, 1);
        ASSERT_FALSE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, sp("(in3[2,3,marker,9])"),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, nullptr));
    }
    {
        // The fixed gate is known: only the marker gate remains.
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileInstruction(f.memory, kTrialKey, andSide, nested);
        gl::upsertStatementKey(f.memory.intKnownStatements,
            gl::packStatementKey(f.memory.nameMap.encode(sp("(in[7,1])")), gl::NameMap::MAIN_ID), true);
        gl::IntegrationTrialRec record;
        ASSERT_TRUE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, &record));
        ASSERT_EQ(record.groupEnd[0], 1);
    }
    {
        // No gate element at all, or only a known fixed one: ungated.
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileInstruction(f.memory, kTrialKey, { "(implication58[u_6,marker])", "(in2[u_5,x,marker])" });
        ASSERT_FALSE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, nullptr));
    }
    {
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileInstruction(f.memory, kTrialKey, { "(in[u_7,u_1])", "(implication58[u_6,marker])" });
        gl::upsertStatementKey(f.memory.intKnownStatements,
            gl::packStatementKey(f.memory.nameMap.encode(sp("(in[7,1])")), gl::NameMap::MAIN_ID), true);
        ASSERT_FALSE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, nullptr));
    }
}

// The matched key is the admission premise, never a gate: an and element
// equal to it is skipped, and an and entity equal to it is not walked.
TEST(integration_trial, matched_key_is_never_a_gate) {
    {
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileInstruction(f.memory, kTrialKey, { kTrialKey, "(implication58[u_6,marker])" });
        ASSERT_FALSE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, nullptr));
    }
    {
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileEntities(f.memory, kTrialKey, {
            { "existence", "(existence55[u_2,u_3,u_4,u_5,u_6])", 5, { kTrialKey, "(and56[u_5,u_6,marker])" } },
            { "and", kTrialKey.c_str(), 4, { "(in[u_7,u_1])" } },
            { "and", "(and56[u_5,u_6,marker])", 3, { "(implication58[u_6,marker])" } } });
        ASSERT_FALSE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("w1"), nullptr, 0, nullptr, 0, nullptr));
    }
}

// Another new name of the admission takes on the gates of the existence
// beside the key's existence whose bound variable it fills: the filler, with
// the marker read as the matched witness and the variable as the name, must
// be one of the facts. The matched witness alone is ungated here (the key's
// existence has no and beside the key).
TEST(integration_trial, coadmission_gates_come_from_the_filled_existence) {
    TrialFixture f;
    f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
    fileCoadmissionInstruction(f.memory, kTrialKey,
        { "(in2[u_5,u_6,pi_1])", "(implication58[u_6,pi_1])", "(in2[u_5,marker,pi_1])", "(in[u_7,u_1])" });
    const gl::StrSpan facts[2] = { sp("(in3[2,3,wI,4])"), sp("(in2[wB,wI,2])") };
    const gl::StrSpan others[1] = { sp("wB") };
    auto& gates = f.analyzer.admissionMemory;
    ASSERT_TRUE(gates.integrationTrialGates(f.memory, gl::StrSpan(f.marker), sp("main"), sp("wI"),
        others, 1, facts, 2, nullptr));
    ASSERT_TRUE(f.memory.nameMap.lookup(sp("(in2[5,6,wB])")) == 0);
    gl::IntegrationTrialRec record;
    ASSERT_TRUE(gates.integrationTrialGates(f.memory, gl::StrSpan(f.marker), sp("main"), sp("wI"),
        others, 1, facts, 2, &record));
    ASSERT_EQ(record.groupCount, 1);
    ASSERT_EQ(record.groupEnd[0], 2);
    ASSERT_EQ(record.gates[0], f.memory.nameMap.lookup(sp("(in2[5,6,wB])")));
    ASSERT_EQ(record.gates[1], f.memory.nameMap.lookup(sp("(in[7,1])")));
    // The name fills nothing without the filler fact: ungated.
    const gl::StrSpan unrelated[1] = { sp("(in3[2,3,wI,4])") };
    ASSERT_FALSE(gates.integrationTrialGates(f.memory, gl::StrSpan(f.marker), sp("main"), sp("wI"),
        others, 1, unrelated, 1, nullptr));
    // The matched witness alone has no gate.
    ASSERT_FALSE(gates.integrationTrialGates(f.memory, gl::StrSpan(f.marker), sp("main"), sp("wI"),
        nullptr, 0, nullptr, 0, nullptr));
}

// A new name filling a level above the key takes on the gates of that level:
// the gate elements for its variable in the ands of the entity whose text it
// fills. Without such an element the name comes along ungated.
TEST(integration_trial, name_above_the_key_takes_the_gates_of_its_level) {
    const gl::StrSpan facts[2] = { sp("(in3[2,3,wI,4])"), sp("(in2[wB,wI,2])") };
    const gl::StrSpan others[1] = { sp("wB") };
    {
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileEntities(f.memory, kTrialKey, {
            { "existence", "(existence61[u_2,u_3,u_4,pi_1])", 4, { kTrialKey, "(and64[u_2,u_3,pi_1])" } },
            { "and", "(and64[u_2,u_3,pi_1])", 3, { "(in2[pi_1,marker,u_2])", "(in[u_3,pi_1])" } } });
        gl::IntegrationTrialRec record;
        ASSERT_TRUE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker), sp("main"),
            sp("wI"), others, 1, facts, 2, &record));
        ASSERT_EQ(record.groupCount, 1);
        ASSERT_EQ(record.groupEnd[0], 1);
        ASSERT_EQ(record.gates[0], f.memory.nameMap.lookup(sp("(in[3,wB])")));
    }
    {
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileEntities(f.memory, kTrialKey, {
            { "existence", "(existence61[u_2,u_3,u_4,pi_1])", 4, { kTrialKey, "(and64[u_2,u_3,pi_1])" } },
            { "and", "(and64[u_2,u_3,pi_1])", 3, { "(in2[pi_1,marker,u_2])", "(implication58[u_3,pi_1])" } } });
        ASSERT_FALSE(f.analyzer.admissionMemory.integrationTrialGates(f.memory, gl::StrSpan(f.marker), sp("main"),
            sp("wI"), others, 1, facts, 2, nullptr));
    }
}

// A nested witness whose route makes its enclosing witness a new name with
// a gate opens a trial for the source: both new names are minted, the scope
// is named after them in slot order, their facts go to the trial scope, the
// instruction is not committed, and the record holds the gate the enclosing
// witness takes on and lists both names.
TEST(integration_trial, coadmitted_witness_opens_a_trial) {
    TrialFixture f;
    f.defineNestedSource();
    fileCoadmissionInstruction(f.memory, kTrialKey, { "(in2[u_5,u_6,pi_1])", "(implication58[u_6,pi_1])" });
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 2);

    const gl::IntegrationTrialRec* trial = f.memory.lbMemory.integrationTrials.find(f.trialKey());
    ASSERT_TRUE(trial != nullptr);
    ASSERT_EQ(trial->status, gl::IntegrationTrialRec::kOpen);
    std::vector<int32_t> ordinals;
    const std::vector<std::string> names = trialWitnesses(f.memory, *trial, &ordinals);
    ASSERT_EQ(names.size(), size_t{2});
    ASSERT_EQ(ordinals[0], 0);
    // The nested existence has an operator fact: its algebra slot comes
    // first and stays unminted.
    ASSERT_EQ(ordinals[1], 2);
    const std::string scope = f.memory.nameMap.decode(trial->trialScope);
    ASSERT_EQ(scope, "main_boundary_integration_" + names[0] + "_" + names[1]);
    const std::string& witness = names[0];
    ASSERT_EQ(trial->groupCount, 1);
    ASSERT_EQ(trial->groupEnd[0], 1);
    ASSERT_EQ(trial->gates[0], f.memory.nameMap.lookup(gl::StrSpan("(in2[5,6," + witness + "])")));
    ASSERT_TRUE(published(output, "(in3[2,3," + names[1] + ",4])", scope));

    bool atTrial = false, elsewhere = false;
    output.statements.forEachSorted([&](gl::StrSpan text, gl::StrSpan validity) {
        const std::string t(text.ptr, static_cast<size_t>(text.len));
        const std::string v(validity.ptr, static_cast<size_t>(validity.len));
        if (t.find(witness) == std::string::npos) return;
        if (v == scope) atTrial = true;
        else elsewhere = true;
    });
    ASSERT_TRUE(atTrial);
    ASSERT_FALSE(elsewhere);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 0);
}

// A passed trial admits at its parent what it listed: the source's
// expansion gives the listed slots their recorded names without drawing a
// number, the nested witness's route under the key is committed, and every
// listed statement is published at main.
TEST(integration_trial, passed_coadmission_trial_admits_its_recorded_names) {
    TrialFixture f;
    f.defineNestedSource();
    const std::vector<EntitySpec> entities = coadmissionEntities(kTrialKey,
        { "(in2[u_5,u_6,pi_lev_2_1])", "(implicationT58[u_6,pi_lev_2_1])" }, "pi_lev_2_1");
    registerTableEntities(f.analyzer, entities);
    fileEntities(f.memory, kTrialKey, entities);
    f.depositLocal();
    ASSERT_EQ(f.memory.startInt, 2);
    passTrial(f);
    const gl::IntegrationTrialRec trial = *f.memory.lbMemory.integrationTrials.find(f.trialKey());
    const std::vector<std::string> statements = trialStatements(f.memory, trial);
    ASSERT_TRUE(!statements.empty());

    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 2);
    ASSERT_EQ(f.memory.lbMemory.integrationTrials.count(), 1);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 1);
    for (const std::string& statement : statements) ASSERT_TRUE(published(output, statement, "main"));
    const std::vector<std::string> names = trialWitnesses(f.memory, trial);
    ASSERT_TRUE(published(output, "(in3[2,3," + names[1] + ",4])", "main"));
}

// An enclosing witness has a place in the matched instruction above the key
// (a variable of the key's existence's signature or sides, or of an existence
// enclosing it), beside it (the bound variable of an existence on a side of
// the key's existence) or nowhere (D-482). A place counts
// only when the filling text, with the marker read as the nested witness, is
// one of the facts.
TEST(integration_trial, enclosing_fill_reads_the_levels_of_the_instruction) {
    using AM = gl::AdmissionMemory;
    const gl::StrSpan facts[2] = { sp("(in3[2,3,wI,4])"), sp("(in2[wB,wI,2])") };
    {
        // Above: the key's existence carries the variable in its signature
        // and in a side.
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileEntities(f.memory, kTrialKey, {
            { "existence", "(existence61[u_2,u_3,u_4,pi_1])", 4, { kTrialKey, "(in2[pi_1,marker,u_2])" } } });
        ASSERT_EQ(f.analyzer.admissionMemory.integrationEnclosingFill(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("wB"), sp("wI"), facts, 2), AM::kEnclosingFillAbove);
        const gl::StrSpan signatureOnly[2] = { sp("(in3[2,3,wI,4])"), sp("(existence61[2,3,4,wB])") };
        ASSERT_EQ(f.analyzer.admissionMemory.integrationEnclosingFill(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("wB"), sp("wI"), signatureOnly, 2), AM::kEnclosingFillAbove);
    }
    {
        // Above: an existence enclosing the key's existence binds the variable.
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileEntities(f.memory, kTrialKey, {
            { "existence", "(existence62[u_2,u_3,u_4])", 3, { "(in2[pi_2,u_3,u_2])", "(existence63[u_2,u_3,u_4,pi_2])" } },
            { "existence", "(existence63[u_2,u_3,u_4,pi_2])", 4, { kTrialKey, "(in[marker,u_2])" } } });
        const gl::StrSpan outer[2] = { sp("(in3[2,3,wI,4])"), sp("(in2[wB,3,2])") };
        ASSERT_EQ(f.analyzer.admissionMemory.integrationEnclosingFill(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("wB"), sp("wI"), outer, 2), AM::kEnclosingFillAbove);
    }
    {
        // Beside: the bound variable of the existence on a side of the key's
        // existence, filled through its filler side.
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileCoadmissionInstruction(f.memory, kTrialKey, { "(in2[u_5,u_6,pi_1])" });
        ASSERT_EQ(f.analyzer.admissionMemory.integrationEnclosingFill(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("wB"), sp("wI"), facts, 2), AM::kEnclosingFillBeside);
        // Without the filler fact the witness has no place.
        const gl::StrSpan unrelated[1] = { sp("(in3[2,3,wI,4])") };
        ASSERT_EQ(f.analyzer.admissionMemory.integrationEnclosingFill(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("wB"), sp("wI"), unrelated, 1), AM::kEnclosingFillNone);
    }
    {
        // Nowhere: the instruction has no variable for the witness.
        TrialFixture f;
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileInstruction(f.memory, kTrialKey, { "(in[u_7,u_1])" });
        ASSERT_EQ(f.analyzer.admissionMemory.integrationEnclosingFill(f.memory, gl::StrSpan(f.marker),
            sp("main"), sp("wB"), sp("wI"), facts, 2), AM::kEnclosingFillNone);
    }
}

// A nested witness matched by an integration key admits nothing beyond the
// levels of its instruction (D-482): when its enclosing
// witness has no place there and no route of its own, neither is minted, the
// instruction is not committed, and the source is parked under the enclosing
// witness's construction, not under the nested witness's present demand.
TEST(integration_trial, nested_witness_waits_for_an_enclosing_witness_without_a_level) {
    TrialFixture f;
    f.defineNestedSource();
    fileInstruction(f.memory, kTrialKey, { "(in[u_7,u_1])" });
    gl::DisintProducts output(&f.arena);
    f.disintegrate("main", false, output);
    ASSERT_EQ(f.memory.startInt, 0);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 0);
    const int64_t nestedPark = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        std::string("(in3[2,3,marker,4])"), std::string("main"));
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.lookup(nestedPark) == 0);
    const int64_t enclosingPark = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        std::string("(existence98[marker,2,3,4])"), std::string("main"));
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.lookup(enclosingPark) != 0);
}

// When the enclosing witness is admitted through its own route, the nested
// witness follows it in the same expansion under its key.
TEST(integration_trial, nested_witness_follows_its_admitted_enclosing_witness) {
    TrialFixture f;
    f.defineNestedSource();
    fileRegisteredInstruction(f.analyzer, f.memory, kTrialKey, { "(in[u_5,u_6])" });
    f.memory.overallHashMemory.admissionSetIntegration.mint(gl::mintTemplateKey(f.memory.templateInterner,
        f.memory.nameMap, std::string("(existence98[marker,2,3,4])"), std::string("main")));
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 2);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 1);
}

// A witness whose only route is a gated instruction is minted, but its facts
// go to its own trial scope with the source's zoom into it; at main the
// instruction is not committed and no receipt or integration park is written.
// A second disintegration finds the open trial and adds nothing.
TEST(integration_trial, gated_route_opens_a_trial_scope) {
    TrialFixture f;
    fileInstruction(f.memory, kTrialKey, { "(in2[u_5,u_6,marker])", "(implication58[u_6,marker])" });
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 1);

    const gl::IntegrationTrialRec* trial = f.memory.lbMemory.integrationTrials.find(f.trialKey());
    ASSERT_TRUE(trial != nullptr);
    ASSERT_EQ(trial->status, gl::IntegrationTrialRec::kOpen);
    ASSERT_TRUE(gl::ScopeSyntax::isIntegrationTrialScope(f.memory.nameMap, trial->trialScope));
    ASSERT_EQ(f.memory.nameMap.parentOf(trial->trialScope), gl::NameMap::MAIN_ID);
    ASSERT_EQ(f.memory.nameMap.decode(trial->source), f.source);
    ASSERT_EQ(trial->flags & 2, 2);
    const std::string scope = f.memory.nameMap.decode(trial->trialScope);
    std::vector<int32_t> ordinals;
    const std::vector<std::string> names = trialWitnesses(f.memory, *trial, &ordinals);
    ASSERT_EQ(names.size(), size_t{1});
    ASSERT_EQ(ordinals[0], 0);
    ASSERT_EQ(scope, "main_boundary_integration_" + names[0]);
    const std::string& witness = names[0];
    ASSERT_EQ(trial->groupCount, 1);
    ASSERT_EQ(trial->gates[0], f.memory.nameMap.lookup(gl::StrSpan("(in2[5,6," + witness + "])")));
    const std::vector<std::string> statements = trialStatements(f.memory, *trial);
    ASSERT_TRUE(!statements.empty());
    for (const std::string& statement : statements) ASSERT_TRUE(published(output, statement, scope));

    bool atTrial = false, elsewhere = false;
    output.statements.forEachSorted([&](gl::StrSpan text, gl::StrSpan validity) {
        const std::string t(text.ptr, static_cast<size_t>(text.len));
        const std::string v(validity.ptr, static_cast<size_t>(validity.len));
        if (t.find(witness) == std::string::npos) return;
        if (v == scope) atTrial = true;
        else elsewhere = true;
    });
    ASSERT_TRUE(atTrial);
    ASSERT_FALSE(elsewhere);

    const int32_t* receipt = f.memory.existenceReplays.find(f.trialKey());
    ASSERT_TRUE(receipt == nullptr || (*receipt & 2) == 0);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.empty());
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 0);

    bool zoomed = false;
    for (const auto& entry : gl::decodeOriginMapSorted(f.memory.exprOriginMap, f.memory.originInterner)) {
        if (entry.first.first != f.source || entry.first.second != scope) continue;
        for (const auto& line : entry.second)
            if (line.first == "zoom in" && line.second.size() == 1
                && line.second[0].original == f.source && line.second[0].validityName == "main")
                zoomed = true;
    }
    ASSERT_TRUE(zoomed);

    gl::DisintProducts again(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, again));
    ASSERT_EQ(f.memory.startInt, 1);
    ASSERT_EQ(f.memory.lbMemory.integrationTrials.count(), 1);
}

// The admission set (an ungated route) admits the witness at main as usual:
// no trial scope or record.
TEST(integration_trial, ungated_route_admits_as_usual) {
    TrialFixture f;
    fileRegisteredInstruction(f.analyzer, f.memory, kTrialKey, { "(in2[u_5,u_6,marker])" });
    f.demand("main");
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 1);
    ASSERT_TRUE(f.memory.lbMemory.integrationTrials.empty());
    bool atMain = false;
    output.statements.forEachSorted([&](gl::StrSpan text, gl::StrSpan validity) {
        if (gl::containsSpan(text, sp("int_")) && gl::equalSpans(validity, sp("main"))) atMain = true;
    });
    ASSERT_TRUE(atMain);
    const int32_t* receipt = f.memory.existenceReplays.find(f.trialKey());
    ASSERT_TRUE(receipt != nullptr && (*receipt & 2) != 0);
}

// The shared replay tail marks the replayed kinds pending in the target's
// receipt, mails the source there and cites it by zoom from its park scope.
TEST(integration_trial, source_replay_marks_pending_and_mails_the_source) {
    TrialFixture f;
    const std::string child = f.scope(gl::NameMap::MAIN_ID, "child");
    const int levels[2] = { 0, 2 };
    f.analyzer.rejectedSourceReplay.queueSourceReplay(f.memory, gl::StrSpan(f.source), sp("main"),
        gl::StrSpan(child), 2, 4, 3, levels, 2, 1);
    ASSERT_EQ(f.receipt(child), 8);
    ASSERT_EQ(f.memory.sameIterationInternalMail.statements_.count(), 1);
    ASSERT_FALSE(f.memory.sameIterationInternalMail.origins_.empty());
    // A relay source also marks the relay bit.
    TrialFixture g;
    const std::string other = g.scope(gl::NameMap::MAIN_ID, "child");
    g.analyzer.rejectedSourceReplay.queueSourceReplay(g.memory, gl::StrSpan(g.source), sp("main"),
        gl::StrSpan(other), 2, 8, 0, levels, 2, 0);
    ASSERT_EQ(g.receipt(other), 8 | 16);
}

// A passed trial's source is replayed at the trial's parent once; while that
// replay is pending, no second one is queued.
TEST(integration_trial, trial_source_replays_once_at_the_parent) {
    TrialFixture f;
    fileInstruction(f.memory, kTrialKey, { "(in2[u_5,u_6,marker])", "(implication58[u_6,marker])" });
    f.depositLocal();
    const gl::IntegrationTrialRec* open = f.memory.lbMemory.integrationTrials.find(f.trialKey());
    ASSERT_TRUE(open != nullptr);
    gl::IntegrationTrialRec trial = *open;
    trial.status = gl::IntegrationTrialRec::kPassed;
    const int32_t mailed = f.memory.sameIterationInternalMail.statements_.count();
    ASSERT_TRUE(f.analyzer.rejectedSourceReplay.replayTrialSource(f.memory, trial));
    ASSERT_EQ(f.memory.sameIterationInternalMail.statements_.count(), mailed + 1);
    ASSERT_EQ(f.receipt("main") & 8, 8);
    ASSERT_FALSE(f.analyzer.rejectedSourceReplay.replayTrialSource(f.memory, trial));
    ASSERT_EQ(f.memory.sameIterationInternalMail.statements_.count(), mailed + 1);
}

// The full trial cycle: the deposit opens a trial; while its gate is unknown
// the drain leaves it open; once the gate is known at the trial scope the
// drain passes it, replays the source at main and stages the trial scope's
// wipe; the absorb admits the witness at main under the name the trial minted
// and commits the instruction, and the passed record stays as receipt.
TEST(integration_trial, known_gate_promotes_the_witness_to_the_parent) {
    TrialFixture f;
    fileRegisteredInstruction(f.analyzer, f.memory, kTrialKey,
        { "(in2[u_5,u_6,marker])", "(implicationT58[u_6,marker])" });
    f.depositLocal();
    ASSERT_EQ(f.memory.startInt, 1);
    const gl::IntegrationTrialRec* trial = f.memory.lbMemory.integrationTrials.find(f.trialKey());
    ASSERT_TRUE(trial != nullptr);
    const gl::NameId trialScope = trial->trialScope;
    const std::string gate = f.memory.nameMap.decode(trial->gates[0]);
    const std::string scope = f.memory.nameMap.decode(trialScope);
    const std::string witness = trialWitnesses(f.memory, *trial)[0];
    ASSERT_EQ(scope, "main_boundary_integration_" + witness);
    ASSERT_TRUE(gl::lookupStatementFlags(f.memory.intKnownStatements, f.memory.nameMap,
        gl::StrSpan("(in3[2,3," + witness + ",4])"), gl::StrSpan(scope)) != nullptr);

    f.analyzer.implicationSubproofs.drainIntegrationTrials(f.memory);
    ASSERT_EQ(f.memory.lbMemory.integrationTrials.find(f.trialKey())->status, gl::IntegrationTrialRec::kOpen);
    ASSERT_TRUE(f.memory.pendingWipeScopes.empty());

    gl::upsertStatementKey(f.memory.intKnownStatements,
        gl::packStatementKey(f.memory.nameMap.encode(gl::StrSpan(gate)), trialScope), true);
    f.analyzer.implicationSubproofs.drainIntegrationTrials(f.memory);
    ASSERT_EQ(f.memory.lbMemory.integrationTrials.find(f.trialKey())->status, gl::IntegrationTrialRec::kPassed);
    ASSERT_TRUE(f.memory.pendingWipeScopes.contains(trialScope));
    ASSERT_TRUE(f.memory.intValidityNamesToFilter.contains(trialScope));
    ASSERT_EQ(f.receipt("main") & 8, 8);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 0);

    f.absorb();
    ASSERT_EQ(f.memory.startInt, 1);
    ASSERT_TRUE(gl::lookupStatementFlags(f.memory.intKnownStatements, f.memory.nameMap,
        gl::StrSpan("(in3[2,3," + witness + ",4])"), sp("main")) != nullptr);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 1);
    ASSERT_EQ(f.receipt("main") & 2, 2);
    ASSERT_TRUE(f.memory.pendingWipeScopes.empty());
    ASSERT_EQ(f.memory.lbMemory.integrationTrials.count(), 1);
    ASSERT_EQ(f.memory.lbMemory.integrationTrials.find(f.trialKey())->status, gl::IntegrationTrialRec::kPassed);
}

// A passed trial is the scope's receipt for the gate: the gated instruction
// admits the witness at main under the name the trial minted and is committed
// for it; the source's construction is complete there.
TEST(integration_trial, passed_trial_admits_its_witness_under_its_name) {
    TrialFixture f;
    fileRegisteredInstruction(f.analyzer, f.memory, kTrialKey,
        { "(in2[u_5,u_6,marker])", "(implicationT58[u_6,marker])" });
    f.depositLocal();
    ASSERT_EQ(f.memory.startInt, 1);
    passTrial(f);
    const std::string witness = trialWitnesses(f.memory, *f.memory.lbMemory.integrationTrials.find(f.trialKey()))[0];
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 1);
    ASSERT_EQ(f.memory.lbMemory.integrationTrials.count(), 1);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 1);
    ASSERT_TRUE(published(output, "(in3[2,3," + witness + ",4])", "main"));
    const int32_t* receipt = f.memory.existenceReplays.find(f.trialKey());
    ASSERT_TRUE(receipt != nullptr && (*receipt & 2) != 0);
}

// A name the admission creates below the matched witness takes on the gates
// of its place too: the matched witness w has no gate of its own, but its
// admission also creates the counterexample witness x of the negative
// implication in w's source, and x fills the bound variable of the existence
// beside the key, whose and gates it. The trial mints both and lists them in
// slot order.
TEST(integration_trial, nested_name_on_the_other_side_gates_the_admission) {
    TrialFixture f;
    gl::CompiledDefinitionFixture::compiledExpressions(f.analyzer.definitions)["existence99"] =
        gl::LogicalEntity("existence", {"(in3[u_2,u_3,w,u_4])", "!(implication100[w,u_2])"},
            "(existence99[u_1,u_2,u_3,u_4])", 4);
    gl::CompiledDefinitionFixture::compiledExpressions(f.analyzer.definitions)["implication100"] =
        gl::LogicalEntity("implication", {"(in2[x,u_1,u_2])", "(=[x,u_2])"}, "(implication100[u_1,u_2])", 2);
    f.analyzer.supportingRuleCompiler.compileDisintegrationPackages();
    f.source = "(existence99[1,2,3,4])";
    fileCoadmissionInstruction(f.memory, kTrialKey, { "(in2[u_5,u_6,pi_1])" });
    gl::DisintProducts output(&f.arena);
    // Not fully covered: w's operator fact also gives it an algebra twin,
    // whose own counterexample witness stays unresolved.
    f.disintegrate("main", false, output);
    const gl::IntegrationTrialRec* trial = f.memory.lbMemory.integrationTrials.find(f.trialKey());
    ASSERT_TRUE(trial != nullptr);
    ASSERT_EQ(f.memory.startInt, 2);
    std::vector<int32_t> ordinals;
    const std::vector<std::string> names = trialWitnesses(f.memory, *trial, &ordinals);
    ASSERT_EQ(names.size(), size_t{2});
    ASSERT_EQ(ordinals[0], 0);
    ASSERT_EQ(ordinals[1], 1);
    ASSERT_EQ(trial->groupCount, 1);
    ASSERT_EQ(trial->groupEnd[0], 1);
    ASSERT_EQ(trial->gates[0], f.memory.nameMap.lookup(gl::StrSpan("(in2[5,6," + names[1] + "])")));
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 0);
}

// A recorded name is given to a slot without drawing a number; the slot is
// named once, below a named enclosing slot.
TEST(integration_trial, materialize_as_names_a_slot_without_drawing_a_number) {
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    gl::TemporaryWitnessStore store(arena, 0);
    const int32_t outer = store.append(gl::TemporaryWitnessKind::Integration, 0, 0, 0);
    const int32_t inner = store.append(gl::TemporaryWitnessKind::Integration, 0, outer, 1);
    int next = 7;
    ASSERT_TRUE(gl::equalSpans(store.materializeAs(outer, sp("int_0_lev_2_3")), sp("int_0_lev_2_3")));
    ASSERT_TRUE(gl::equalSpans(store.permanent(outer), sp("int_0_lev_2_3")));
    ASSERT_TRUE(gl::equalSpans(store.materialize(outer, 2, next), sp("int_0_lev_2_3")));
    ASSERT_EQ(next, 7);
    ASSERT_TRUE(gl::equalSpans(store.materializeAs(inner, sp("int_0_lev_2_4")), sp("int_0_lev_2_4")));
    ASSERT_EQ(store.lookup(sp("int_0_lev_2_4")), 0);
    ASSERT_EQ(next, 7);
}

// The top witness v's only route is a gated instruction on `in3[2,3,v,4]`,
// which opens its trial. A double-marker key on `in2[marker1,marker2,u_2]`
// gives the slot w nested in v's integration slot a route-5 hit, and
// `in2[v,w,2]` links w to v: that hit is an ungated admitting route of v, so
// no trial opens and v is admitted at main
// (D-501, D-481).
TEST(integration_trial, linked_double_marker_route_below_opens_no_trial) {
    for (const bool doubleKey : { false, true }) {
        TrialFixture f;
        auto& compiled = gl::CompiledDefinitionFixture::compiledExpressions(f.analyzer.definitions);
        compiled["existence97"] = gl::LogicalEntity("existence",
            {"(in3[u_2,u_3,v,u_4])", "(existence98[v,u_2,u_3,u_4])"}, "(existence97[u_1,u_2,u_3,u_4])", 4);
        compiled["existence98"] = gl::LogicalEntity("existence",
            {"(in2[u_1,w,u_2])"}, "(existence98[u_1,u_2,u_3,u_4])", 4);
        f.analyzer.supportingRuleCompiler.compileDisintegrationPackages();
        f.source = "(existence97[1,2,3,4])";
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
        fileRegisteredInstruction(f.analyzer, f.memory, kTrialKey, { "(in2[u_5,u_6,marker])" });
        if (doubleKey)
            f.memory.overallHashMemory.admissionMapIntegrationDouble.insert(gl::mintTemplateKey(
                f.memory.templateInterner, f.memory.nameMap, std::string("(in2[marker1,marker2,u_2])"),
                std::string("main")), int64_t(0));
        gl::DisintProducts output(&f.arena);
        f.disintegrate("main", false, output);
        ASSERT_EQ(f.memory.lbMemory.integrationTrials.find(f.trialKey()) != nullptr, !doubleKey);
        if (!doubleKey) continue;
        ASSERT_TRUE((f.receipt("main") & 2) != 0);
    }
}
// The payload values of every instruction under @p key at `main`, in stored
// order (witness-link gate tests).
static std::vector<std::vector<std::string>> instructionPayloads(gl::Memory& m, const std::string& key) {
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, std::string("main"));
    gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
    entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
    std::vector<std::vector<std::string>> out;
    for (int32_t i = 0; i < entries.entries.size(); ++i) {
        int32_t ids[gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES];
        const int32_t n = entries.valuesAt(i, ids, gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES);
        std::vector<std::string> values;
        for (int32_t v = 0; v < n; ++v) values.push_back(m.valueInterner.decode(ids[v]));
        out.push_back(values);
    }
    return out;
}

// Every path list stored in the block's integration instructions.
static std::vector<std::string> storedPathLists(gl::Memory& m) {
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    gl::ScratchScope frame(arena);
    auto& ami = m.overallHashMemory.admissionMapIntegration;
    std::vector<std::string> lists;
    for (int32_t id = 1; id <= ami.count(); ++id) {
        const int64_t pk = ami.keyAt(id);
        if (ami.lookup(pk) == 0) continue;
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(ami, pk);
        for (int32_t i = 0; i < entries.entries.size(); ++i)
            if (entries.entries[i].pathVarsId != 0)
                lists.push_back(m.valueInterner.decode(entries.entries[i].pathVarsId));
    }
    return lists;
}

// The instruction of `coadmissionEntities` (its matched witness alone has no
// gate) with the bound variable `pi_lev_2_1`, registered for a commit, plus
// the atomic entities of the key and of the filler `in2[pi_lev_2_1,marker,u_2]`
// as a prepared instruction lists them: once the marker is filled, the
// filler keys the continuation (Case C).
static std::vector<EntitySpec> registeredLinkInstruction(TrialFixture& f) {
    std::vector<EntitySpec> entities = coadmissionEntities(kTrialKey,
        { "(in2[u_5,u_6,pi_lev_2_1])" }, "pi_lev_2_1");
    registerTableEntities(f.analyzer, entities);
    entities.push_back({ "atomic", kTrialKey, 4, {} });
    entities.push_back({ "atomic", "(in2[pi_lev_2_1,marker,u_2])", 3, {} });
    return entities;
}

// In an induction step block an instruction whose path list holds the
// argument the linked source links its witness to (1) refuses the route:
// the deposited source's witness is parked, the instruction records the
// refusal and the park-time rendezvous queues no replay. A new accepting
// instruction under the key reopens the demand, and the wake queues the
// replay.
TEST(witness_link_gate, step_block_parks_a_witness_linked_to_its_path) {
    TrialFixture f;
    f.defineLinkedSource();
    f.memory.recursionAuxyImplId = 1;
    const std::vector<EntitySpec> entities = registeredLinkInstruction(f);
    fileEntities(f.memory, kTrialKey, entities, "[1,9]");
    f.depositLocal();
    ASSERT_EQ(f.memory.startInt, 0);
    const int64_t park = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        std::string("(in3[2,3,marker,4])"), std::string("main"));
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.lookup(park) != 0);
    const std::vector<std::vector<std::string>> values = instructionPayloads(f.memory, kTrialKey);
    ASSERT_EQ(values.size(), std::size_t{1});
    ASSERT_EQ(values[0].size(), std::size_t{1});
    ASSERT_EQ(values[0][0], std::string("refused:(existence93[1,2,3,4,5])"));
    ASSERT_EQ(f.receipt("main") & 8, 0);

    fileEntities(f.memory, kTrialKey, entities, "[9]");
    f.wake("main");
    ASSERT_EQ(f.receipt("main") & 8, 8);
}

// An instruction whose list does not hold the linked argument admits the
// witness in a step block, and the continuation's list gains the witness.
TEST(witness_link_gate, step_block_admits_a_witness_linked_off_its_path) {
    TrialFixture f;
    f.defineLinkedSource();
    f.memory.recursionAuxyImplId = 1;
    fileEntities(f.memory, kTrialKey, registeredLinkInstruction(f), "[9]");
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 1);
    const std::vector<std::vector<std::string>> values = instructionPayloads(f.memory, kTrialKey);
    ASSERT_EQ(values.size(), std::size_t{1});
    ASSERT_EQ(values[0].size(), std::size_t{1});
    const std::string& witness = values[0][0];
    ASSERT_TRUE(witness.rfind("int_", 0) == 0);
    const std::string extended = "[" + std::min(witness, std::string("9")) + ","
        + std::max(witness, std::string("9")) + "]";
    const std::vector<std::string> lists = storedPathLists(f.memory);
    ASSERT_TRUE(std::find(lists.begin(), lists.end(), extended) != lists.end());
}

// Beside a refusing instruction an accepting one admits the witness; only
// the accepting instruction is committed.
TEST(witness_link_gate, accepting_instruction_admits_beside_a_refusing_one) {
    TrialFixture f;
    f.defineLinkedSource();
    f.memory.recursionAuxyImplId = 1;
    const std::vector<EntitySpec> entities = registeredLinkInstruction(f);
    fileEntities(f.memory, kTrialKey, entities, "[1,9]");
    fileEntities(f.memory, kTrialKey, entities, "[9]");
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 1);
    const std::vector<std::vector<std::string>> values = instructionPayloads(f.memory, kTrialKey);
    ASSERT_EQ(values.size(), std::size_t{2});
    ASSERT_TRUE(values[0].empty());
    ASSERT_EQ(values[1].size(), std::size_t{1});
}

// Outside induction step blocks the gate is off: the same list admits.
TEST(witness_link_gate, gate_is_off_outside_step_blocks) {
    TrialFixture f;
    f.defineLinkedSource();
    fileEntities(f.memory, kTrialKey, registeredLinkInstruction(f), "[1,9]");
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate("main", false, output));
    ASSERT_EQ(f.memory.startInt, 1);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 1);
}

// The enclosing fill and the trial gates read only the instructions the
// gate leaves open; outside step blocks every instruction counts.
TEST(witness_link_gate, enclosing_fill_and_trial_gates_skip_refusing_instructions) {
    using AM = gl::AdmissionMemory;
    const gl::StrSpan facts[2] = { sp("(in3[2,3,wI,4])"), sp("(in2[wB,wI,2])") };
    const gl::StrSpan others[1] = { sp("wB") };
    const gl::StrSpan linkedR[1] = { sp("R") };
    const gl::StrSpan linkedA[1] = { sp("A") };
    TrialFixture f;
    f.memory.recursionAuxyImplId = 1;
    f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, gl::NameMap::MAIN_ID);
    fileEntities(f.memory, kTrialKey, coadmissionEntities(kTrialKey, { "(in2[u_5,u_6,pi_1])" }, "pi_1"), "[R]");
    auto& gates = f.analyzer.admissionMemory;
    const gl::StrSpan marker(f.marker);
    ASSERT_EQ(gates.integrationEnclosingFill(f.memory, marker, sp("main"), sp("wB"), sp("wI"), facts, 2,
        linkedA, 1), AM::kEnclosingFillBeside);
    ASSERT_EQ(gates.integrationEnclosingFill(f.memory, marker, sp("main"), sp("wB"), sp("wI"), facts, 2,
        linkedR, 1), AM::kEnclosingFillNone);
    ASSERT_TRUE(gates.integrationTrialGates(f.memory, marker, sp("main"), sp("wI"), others, 1, facts, 2,
        nullptr, linkedA, 1));
    ASSERT_FALSE(gates.integrationTrialGates(f.memory, marker, sp("main"), sp("wI"), others, 1, facts, 2,
        nullptr, linkedR, 1));
    f.memory.recursionAuxyImplId = 0;
    ASSERT_EQ(gates.integrationEnclosingFill(f.memory, marker, sp("main"), sp("wB"), sp("wI"), facts, 2,
        linkedR, 1), AM::kEnclosingFillBeside);
    ASSERT_TRUE(gates.integrationTrialGates(f.memory, marker, sp("main"), sp("wI"), others, 1, facts, 2,
        nullptr, linkedR, 1));
}

// The source's history: an implication whose premise `in2[7,8,s]` names 7 with
// the definition set of existence93's witness w (a number), so an instruction
// whose path list holds 7 is linked to the witness through the derivation tree
// (D-522).
static void depositWithHistory(TrialFixture& f) {
    gl::addOriginEncoded(f.memory.exprOriginMap, f.memory.originInterner, sp("(in2[7,8,s])"), sp("main"),
        gl::OriginTag::taskFormulation, nullptr, 0, 30);
    const gl::OriginDep deps[2] = { { sp("(>[1](in2[1,8,s])(in[1,u_N]))"), sp("main") },
                                    { sp("(in2[7,8,s])"), sp("main") } };
    f.analyzer.statementMemory.addExprToMemoryBlock(gl::StrSpan(f.source), f.memory, 0, 1, f.levels, 2,
        gl::TransientOrigin{true, gl::OriginTag::implication, deps, 2}, -1, -1, sp("main"), false);
}

// In an induction step block a witness that is not an operator output (no
// operator is configured here) is refused through its derivation tree: the
// source's history names 7, which the instruction's path list holds, with the
// witness's definition set. The witness is parked, the instruction records the
// refusal and no replay is queued. Without that history the same instruction
// admits the witness.
TEST(witness_history_gate, step_block_parks_a_witness_whose_history_names_its_path) {
    for (const bool history : { true, false }) {
        TrialFixture f;
        f.defineLinkedSource();
        gl::CompiledDefinitionFixture::operators(f.analyzer.definitions).clear();
        f.memory.recursionAuxyImplId = 1;
        fileEntities(f.memory, kTrialKey, registeredLinkInstruction(f), "[7,9]");
        if (history) depositWithHistory(f);
        else f.depositLocal();
        ASSERT_EQ(f.memory.startInt, history ? 0 : 1);
        if (!history) continue;
        const int64_t park = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
            std::string("(in3[2,3,marker,4])"), std::string("main"));
        ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.lookup(park) != 0);
        const std::vector<std::vector<std::string>> values = instructionPayloads(f.memory, kTrialKey);
        ASSERT_EQ(values.size(), std::size_t{1});
        ASSERT_EQ(values[0].size(), std::size_t{1});
        ASSERT_EQ(values[0][0], std::string("refused:(existence93[1,2,3,4,5])"));
        ASSERT_EQ(f.receipt("main") & 8, 0);
    }
}

// The same history leaves an operator output alone: with `in3` an operator,
// existence93's witness w is in3's output and the instruction admits it.
TEST(witness_history_gate, operator_output_witness_is_exempt_from_its_history) {
    TrialFixture f;
    f.defineLinkedSource();
    f.memory.recursionAuxyImplId = 1;
    fileEntities(f.memory, kTrialKey, registeredLinkInstruction(f), "[7,9]");
    depositWithHistory(f);
    ASSERT_EQ(f.memory.startInt, 1);
}

// The payloads of an OR case and of an induction (catcher) scope, for the
// fixture's `scope`.
static const char* const kOrCasePayload = "ordis_(or1[5,6])_((in[5,6]))";
static const char* const kCatcherPayload = "is_(in3[marker1,marker2,u_9,u_5])_int_1_int_2";

// An integration demand serves its own scope and the induction scopes and OR
// cases whose parent chain reaches its scope through induction scopes only:
// the serving owner is the first scope above them that is not an induction
// scope, and no other scope has one (D-572).
TEST(ancestor_integration_demand, ancestor_demand_owner_matrix) {
    TrialFixture f;
    gl::AdmissionMemory& admission = f.analyzer.admissionMemory;
    const auto id = [&](const std::string& name) { return f.memory.nameMap.lookup(gl::StrSpan(name)); };
    const std::string caseAtMain = f.scope(gl::NameMap::MAIN_ID, kOrCasePayload);
    const std::string catcher = f.scope(gl::NameMap::MAIN_ID, kCatcherPayload);
    const std::string caseInCatcher = f.scope(id(catcher), kOrCasePayload);
    const std::string nestedCatcher = f.scope(id(catcher), kCatcherPayload);
    const std::string caseInNested = f.scope(id(nestedCatcher), kOrCasePayload);
    const std::string caseInCase = f.scope(id(caseAtMain), kOrCasePayload);
    const std::string plain = f.scope(gl::NameMap::MAIN_ID, "case_child");
    const std::string caseInPlain = f.scope(id(plain), kOrCasePayload);
    const std::string catcherInPlain = f.scope(id(plain), kCatcherPayload);
    const std::string catcherInCase = f.scope(id(caseAtMain), kCatcherPayload);
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, gl::NameMap::MAIN_ID), 0);
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(plain)), 0);
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(catcher)), gl::NameMap::MAIN_ID);
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(nestedCatcher)), gl::NameMap::MAIN_ID);
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(catcherInPlain)), id(plain));
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(catcherInCase)), id(caseAtMain));
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(caseAtMain)), gl::NameMap::MAIN_ID);
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(caseInCatcher)), gl::NameMap::MAIN_ID);
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(caseInNested)), gl::NameMap::MAIN_ID);
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(caseInCase)), id(caseAtMain));
    ASSERT_EQ(admission.ancestorDemandOwner(f.memory, id(caseInPlain)), id(plain));
}

// A route-3 demand is visible at its own scope and at the induction scopes and
// OR cases it serves, never at another scope below it; a served scope lists
// its own keys first, then its owner's.
TEST(ancestor_integration_demand, demands_are_visible_at_their_scope_and_the_scopes_they_serve) {
    TrialFixture f;
    const auto id = [&](const std::string& name) { return f.memory.nameMap.lookup(gl::StrSpan(name)); };
    const std::string catcher = f.scope(gl::NameMap::MAIN_ID, kCatcherPayload);
    const std::string served = f.scope(id(catcher), kOrCasePayload);
    const std::string plain = f.scope(gl::NameMap::MAIN_ID, "case_child");
    const std::string unserved = f.scope(id(plain), kOrCasePayload);
    const std::vector<EntitySpec> entities = instructionEntities(kTrialKey, { "(in2[u_5,u_6,marker])" }, {});
    fileEntities(f.memory, kTrialKey, entities);
    gl::AdmissionMemory& admission = f.analyzer.admissionMemory;
    const auto visible = [&](const std::string& marked, const std::string& at) {
        return admission.integrationDemandVisible(f.memory, gl::StrSpan(marked), gl::StrSpan(at), id(at));
    };
    ASSERT_TRUE(visible(kTrialKey, "main"));
    ASSERT_TRUE(visible(kTrialKey, served));
    ASSERT_TRUE(visible(kTrialKey, catcher));
    ASSERT_FALSE(visible(kTrialKey, plain));
    ASSERT_FALSE(visible(kTrialKey, unserved));
    ASSERT_FALSE(visible("(in3[u_2,u_9,marker,u_4])", served));

    fileEntities(f.memory, kTrialKey, entities, std::string(), served);
    const int64_t atMain = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap, kTrialKey,
        std::string("main"));
    const int64_t atServed = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap, kTrialKey, served);
    const auto keys = [&](const std::string& at) {
        gl::ScratchScope frame(f.arena);
        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<int64_t> out(&f.arena, &dirty);
        admission.matchingVisibleIntegrationDemands(f.memory, gl::StrSpan(kTrialKey), gl::StrSpan(at), id(at),
            out, f.arena);
        std::vector<int64_t> result;
        for (int32_t i = 0; i < out.size(); ++i) result.push_back(out[i]);
        return result;
    };
    ASSERT_TRUE(keys(served) == (std::vector<int64_t>{ atServed, atMain }));
    ASSERT_TRUE(keys("main") == std::vector<int64_t>{ atMain });
    ASSERT_TRUE(keys(catcher) == std::vector<int64_t>{ atMain });
    ASSERT_TRUE(keys(unserved).empty());
}

// A gated route-3 demand at main serves a source disintegrated in an OR case
// below a catcher: the trial opens below that case and nothing parks. In a
// plain child scope the same source parks.
TEST(ancestor_integration_demand, ancestor_demand_opens_a_trial_in_an_or_case) {
    TrialFixture f;
    fileInstruction(f.memory, kTrialKey, { "(in2[u_5,u_6,marker])", "(implication58[u_6,marker])" });
    const std::string catcher = f.scope(gl::NameMap::MAIN_ID, kCatcherPayload);
    const std::string served = f.scope(f.memory.nameMap.lookup(gl::StrSpan(catcher)), kOrCasePayload);
    const gl::NameId servedId = f.memory.nameMap.lookup(gl::StrSpan(served));
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate(served, false, output));
    const int64_t key = gl::packOriginKey(f.analyzer.rejectedSources.canonicalExistenceId(f.memory,
        gl::StrSpan(f.source), servedId), servedId);
    const gl::IntegrationTrialRec* trial = f.memory.lbMemory.integrationTrials.find(key);
    ASSERT_TRUE(trial != nullptr);
    ASSERT_EQ(trial->status, gl::IntegrationTrialRec::kOpen);
    ASSERT_EQ(f.memory.nameMap.parentOf(trial->trialScope), servedId);
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.empty());
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 0);

    TrialFixture g;
    fileInstruction(g.memory, kTrialKey, { "(in2[u_5,u_6,marker])", "(implication58[u_6,marker])" });
    const std::string plain = g.scope(gl::NameMap::MAIN_ID, "case_child");
    gl::DisintProducts parked(&g.arena);
    g.disintegrate(plain, false, parked);
    ASSERT_TRUE(g.memory.lbMemory.integrationTrials.empty());
    ASSERT_TRUE(g.memory.overallHashMemory.rejectedMapIntegration.lookup(gl::mintTemplateKey(
        g.memory.templateInterner, g.memory.nameMap, g.marker, plain)) != 0);
}

// A route-3 demand filed at main after a source parked wakes the park in an OR
// case it serves; the parks in a plain child and in an OR case below a plain
// child stay.
TEST(ancestor_integration_demand, new_demand_wakes_the_parks_in_the_or_cases_it_serves) {
    TrialFixture f;
    const auto id = [&](const std::string& name) { return f.memory.nameMap.lookup(gl::StrSpan(name)); };
    const std::string catcher = f.scope(gl::NameMap::MAIN_ID, kCatcherPayload);
    const std::string served = f.scope(id(catcher), kOrCasePayload);
    const std::string plain = f.scope(gl::NameMap::MAIN_ID, "case_child");
    const std::string unserved = f.scope(id(plain), kOrCasePayload);
    // The source arrives in the three scopes before any demand, the OR case
    // below the plain child first (a statement known at a parent is not
    // stored again below it): it parks in each with its history there.
    f.depositLocal(served);
    f.depositLocal(unserved);
    f.depositLocal(plain);
    for (const std::string& at : { served, plain, unserved })
        ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.lookup(gl::mintTemplateKey(
            f.memory.templateInterner, f.memory.nameMap, f.marker, at)) != 0);
    ASSERT_EQ(f.receipt(served) & 8, 0);
    fileEntities(f.memory, kTrialKey, instructionEntities(kTrialKey, { "(in2[u_5,u_6,marker])" }, {}));
    const int32_t mailed = f.memory.sameIterationInternalMail.statements_.count();
    f.analyzer.rejectedSourceReplay.wakeDescendantIntegrationParks(f.memory, gl::StrSpan(f.marker), sp("main"));
    ASSERT_EQ(f.receipt(served) & 8, 8);
    ASSERT_EQ(f.receipt(plain) & 8, 0);
    ASSERT_EQ(f.receipt(unserved) & 8, 0);
    ASSERT_EQ(f.memory.sameIterationInternalMail.statements_.count(), mailed + 1);

    // A wake from the plain child, which serves the unserved case but owns no
    // demand, queues nothing.
    f.analyzer.rejectedSourceReplay.wakeDescendantIntegrationParks(f.memory, gl::StrSpan(f.marker),
        gl::StrSpan(plain));
    ASSERT_EQ(f.memory.sameIterationInternalMail.statements_.count(), mailed + 1);
    ASSERT_EQ(f.receipt(unserved) & 8, 0);
}

// A wildcard key (`pi_` at one non-marker position) is visible to every query
// that agrees with it at the other positions, whatever the query holds at the
// wildcard, once the block records the wildcard token; the matching demands
// list it (D-586).
TEST(two_marker_wildcard, wildcard_key_matches_any_value_at_its_open_argument) {
    TrialFixture f;
    const std::string wildcardKey = "(in3[u_2,pi_,marker,u_4])";
    fileEntities(f.memory, wildcardKey, instructionEntities(wildcardKey, { "(in2[u_5,u_6,marker])" }, {}));
    gl::AdmissionMemory& admission = f.analyzer.admissionMemory;
    const gl::NameId mainId = f.memory.nameMap.lookup(sp("main"));
    const auto visible = [&](const std::string& marked) {
        return admission.integrationDemandVisible(f.memory, gl::StrSpan(marked), sp("main"), mainId);
    };
    // Without the recorded token only the exact text could match.
    ASSERT_FALSE(gl::AdmissionMemory::integrationWildcardFiled(f.memory));
    ASSERT_FALSE(visible("(in3[u_2,u_3,marker,u_4])"));
    f.memory.overallHashMemory.varsInAdmissionMapIntegrationKeys.mint(f.memory.templateInterner.encode(sp("pi_")));
    ASSERT_TRUE(gl::AdmissionMemory::integrationWildcardFiled(f.memory));
    ASSERT_TRUE(visible("(in3[u_2,u_3,marker,u_4])"));
    ASSERT_TRUE(visible("(in3[u_2,u_9,marker,u_4])"));
    ASSERT_FALSE(visible("(in3[u_7,u_3,marker,u_4])"));
    ASSERT_FALSE(visible("(in3[u_2,u_3,u_4,marker])"));

    const int64_t pk = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap, wildcardKey,
        std::string("main"));
    gl::ScratchScope frame(f.arena);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<int64_t> out(&f.arena, &dirty);
    admission.matchingVisibleIntegrationDemands(f.memory, sp("(in3[u_2,u_9,marker,u_4])"), sp("main"), mainId,
        out, f.arena);
    ASSERT_EQ(out.size(), 1);
    ASSERT_TRUE(out[0] == pk);
}

// A new wildcard key wakes the park it admits at its scope; a wildcard key that
// differs at a fixed argument, or one of another predicate, wakes nothing
// (D-586).
TEST(two_marker_wildcard, wildcard_key_wakes_the_parks_it_admits) {
    TrialFixture f;
    f.depositLocal();
    ASSERT_TRUE(f.memory.overallHashMemory.rejectedMapIntegration.lookup(gl::mintTemplateKey(
        f.memory.templateInterner, f.memory.nameMap, f.marker, std::string("main"))) != 0);
    f.memory.overallHashMemory.varsInAdmissionMapIntegrationKeys.mint(f.memory.templateInterner.encode(sp("pi_")));
    const int32_t mailed = f.memory.sameIterationInternalMail.statements_.count();

    const std::string other = "(in3[u_7,pi_,marker,u_4])";
    fileEntities(f.memory, other, instructionEntities(other, { "(in2[u_5,u_6,marker])" }, {}));
    f.analyzer.rejectedSourceReplay.wakeWildcardIntegrationParks(f.memory, sp("(in3[7,pi_,marker,4])"), sp("main"));
    ASSERT_EQ(f.memory.sameIterationInternalMail.statements_.count(), mailed);
    f.analyzer.rejectedSourceReplay.wakeWildcardIntegrationParks(f.memory, sp("(in2[pi_,marker,5])"), sp("main"));
    ASSERT_EQ(f.memory.sameIterationInternalMail.statements_.count(), mailed);

    const std::string wildcardKey = "(in3[u_2,pi_,marker,u_4])";
    fileEntities(f.memory, wildcardKey, instructionEntities(wildcardKey, { "(in2[u_5,u_6,marker])" }, {}));
    f.analyzer.rejectedSourceReplay.wakeWildcardIntegrationParks(f.memory, sp("(in3[2,pi_,marker,4])"), sp("main"));
    ASSERT_EQ(f.memory.sameIterationInternalMail.statements_.count(), mailed + 1);
}

// An existence part whose signature arguments are all fixed has its fixed
// witness among them only when a name doubles as both: then the integration
// stops (D-590). A distinct fixed witness,
// a bound witness and a part with an open argument do not count.
TEST(doubled_existence_witness, fixed_witness_among_its_own_arguments_is_detected) {
    TrialFixture f;
    gl::ScratchScope frame(f.arena);
    const auto doubled = [&](const std::string& signature, const std::vector<std::string>& elements) {
        gl::WorkInstruction wi(&f.arena);
        const int32_t mark = wi.elemMark();
        for (const std::string& element : elements) wi.addElement(gl::StrSpan(element));
        wi.commitEntity(sp("existence"), gl::StrSpan(signature), sp(""), 4, mark);
        wi.setMarkedGoal(sp("(goal54[1,2,3,4,5,6])"));
        return gl::tests::IntegrationProcessorFixture::witnessDoublesAsArgument(f.analyzer.integrationProcessor, wi);
    };
    // existence96[u_1,u_2,u_3,u_4]: (in[w,u_1]) and (in3[u_2,u_3,w,u_4]).
    ASSERT_TRUE(doubled("(existence96[u_1,u_2,u_3,u_4])", { "(in[u_3,u_1])", "(in3[u_2,u_3,u_3,u_4])" }));
    ASSERT_FALSE(doubled("(existence96[u_1,u_2,u_3,u_4])", { "(in[u_9,u_1])", "(in3[u_2,u_3,u_9,u_4])" }));
    ASSERT_FALSE(doubled("(existence96[u_1,u_2,u_3,u_4])", { "(in[pi_lev_2_0,u_1])", "(in3[u_2,u_3,pi_lev_2_0,u_4])" }));
    ASSERT_FALSE(doubled("(existence96[u_1,pi_lev_2_1,u_3,u_4])", { "(in[u_3,u_1])", "(in3[pi_lev_2_1,u_3,u_3,u_4])" }));
}

// A gate-free route-3 demand at main admits a source at once in a scope it
// serves, while the source parks in a plain child scope. Before a hypothesis
// shot on the path the admission leaves main's instruction untouched; after a
// shot at the catcher between the case and main it records the witness there.
TEST(ancestor_integration_demand, served_scopes_admit_and_update_only_after_a_shot) {
    TrialFixture g;
    g.defineLinkedSource();
    fileEntities(g.memory, kTrialKey, registeredLinkInstruction(g));
    const std::string plain = g.scope(gl::NameMap::MAIN_ID, "case_child");
    gl::DisintProducts parked(&g.arena);
    ASSERT_FALSE(g.disintegrate(plain, false, parked));
    ASSERT_EQ(g.memory.startInt, 0);
    ASSERT_EQ(committedValues(g.memory, kTrialKey), 0);
    ASSERT_TRUE(g.memory.overallHashMemory.rejectedMapIntegration.lookup(gl::mintTemplateKey(
        g.memory.templateInterner, g.memory.nameMap, g.marker, plain)) != 0);

    TrialFixture h;
    h.defineLinkedSource();
    fileEntities(h.memory, kTrialKey, registeredLinkInstruction(h));
    const std::string catcher = h.scope(gl::NameMap::MAIN_ID, kCatcherPayload);
    const std::string served = h.scope(h.memory.nameMap.lookup(gl::StrSpan(catcher)), kOrCasePayload);
    gl::DisintProducts admitted(&h.arena);
    ASSERT_TRUE(h.disintegrate(served, false, admitted));
    ASSERT_EQ(h.memory.startInt, 1);
    ASSERT_TRUE(h.memory.lbMemory.integrationTrials.empty());
    ASSERT_EQ(committedValues(h.memory, kTrialKey), 0);

    TrialFixture s;
    s.defineLinkedSource();
    fileEntities(s.memory, kTrialKey, registeredLinkInstruction(s));
    const std::string shotCatcher = s.scope(gl::NameMap::MAIN_ID, kCatcherPayload);
    const gl::NameId shotCatcherId = s.memory.nameMap.lookup(gl::StrSpan(shotCatcher));
    const std::string shotCase = s.scope(shotCatcherId, kOrCasePayload);
    s.memory.lbMemory.hypothesisShots.mint(shotCatcherId);
    gl::DisintProducts updated(&s.arena);
    ASSERT_TRUE(s.disintegrate(shotCase, false, updated));
    ASSERT_EQ(s.memory.startInt, 1);
    ASSERT_EQ(committedValues(s.memory, kTrialKey), 1);
}

// The shot check walks from the served scope up to, not including, the owner:
// a shot at the scope or at a scope between counts, one at the owner or beside
// the path does not.
TEST(ancestor_integration_demand, hypothesis_shot_on_path_matrix) {
    TrialFixture f;
    gl::AdmissionMemory& admission = f.analyzer.admissionMemory;
    const auto id = [&](const std::string& name) { return f.memory.nameMap.lookup(gl::StrSpan(name)); };
    const std::string catcher = f.scope(gl::NameMap::MAIN_ID, kCatcherPayload);
    const std::string served = f.scope(id(catcher), kOrCasePayload);
    const std::string sibling = f.scope(id(catcher), kCatcherPayload);
    const gl::NameId mainId = gl::NameMap::MAIN_ID;
    ASSERT_FALSE(admission.hypothesisShotOnPath(f.memory, id(served), mainId));
    f.memory.lbMemory.hypothesisShots.mint(mainId);
    f.memory.lbMemory.hypothesisShots.mint(id(sibling));
    ASSERT_FALSE(admission.hypothesisShotOnPath(f.memory, id(served), mainId));
    f.memory.lbMemory.hypothesisShots.mint(id(catcher));
    ASSERT_TRUE(admission.hypothesisShotOnPath(f.memory, id(served), mainId));
    ASSERT_TRUE(admission.hypothesisShotOnPath(f.memory, id(catcher), mainId));
    ASSERT_FALSE(admission.hypothesisShotOnPath(f.memory, id(served), id(catcher)));
    f.memory.lbMemory.hypothesisShots.mint(id(served));
    ASSERT_TRUE(admission.hypothesisShotOnPath(f.memory, id(served), id(catcher)));
}

// The statement door records a statement fired by the induction hypothesis
// (`recursionHead`) as a shot at its scope, whatever the scope's kind.
TEST(ancestor_integration_demand, statement_door_records_a_hypothesis_shot) {
    TrialFixture f;
    const std::string served = f.scope(gl::NameMap::MAIN_ID, kOrCasePayload);
    const gl::NameId servedId = f.memory.nameMap.lookup(gl::StrSpan(served));
    f.analyzer.statementMemory.addExprToMemoryBlock(gl::StrSpan("(in[5,6])"), f.memory, 0, 1, f.levels, 2,
        gl::TransientOrigin{true, gl::OriginTag::taskFormulation, nullptr, 0}, -1, -1, sp("main"), false);
    ASSERT_TRUE(f.memory.lbMemory.hypothesisShots.empty());
    f.analyzer.statementMemory.addExprToMemoryBlock(gl::StrSpan("(in[5,7])"), f.memory, 0, 1, f.levels, 2,
        gl::TransientOrigin{true, gl::OriginTag::taskFormulation, nullptr, 0}, -1, -1, gl::StrSpan(served), false,
        false, nullptr, /*recursionHead=*/true);
    ASSERT_EQ(f.memory.lbMemory.hypothesisShots.count(), 1);
    ASSERT_TRUE(f.memory.lbMemory.hypothesisShots.lookup(servedId) != 0);
}

// After a shot, a commit served by an ancestor's demand continues the
// instruction at the OR case as at its owner: the root goal, prepared at
// main, counts as prepared there and is not prepared again with its witness
// fixed, while the existence the new witness completes is prepared at the
// case.
TEST(ancestor_integration_demand, continuation_counts_the_preparations_above_its_scope) {
    TrialFixture f;
    f.defineLinkedSource();
    const std::string root = "(existence55[2,3,4,5,6])";
    fileEntities(f.memory, kTrialKey, registeredLinkInstruction(f), std::string(), std::string("main"), root);
    f.memory.integrationPrepared.mint(gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap, root,
        std::string("main")));
    const std::string served = f.scope(gl::NameMap::MAIN_ID, kOrCasePayload);
    // The hypothesis shot in the case: the commit updates main's instruction.
    f.memory.lbMemory.hypothesisShots.mint(f.memory.nameMap.lookup(gl::StrSpan(served)));
    gl::DisintProducts output(&f.arena);
    ASSERT_TRUE(f.disintegrate(served, false, output));
    ASSERT_EQ(f.memory.startInt, 1);
    ASSERT_EQ(committedValues(f.memory, kTrialKey), 1);
    std::vector<std::string> preparedAtCase;
    for (int32_t i = 1; i <= f.memory.integrationPrepared.count(); ++i) {
        gl::StrSpan payload, scope;
        gl::decodeTemplateKeyView(f.memory.integrationPrepared.decode(i), f.memory.templateInterner,
            f.memory.nameMap, payload, scope);
        if (gl::equalSpans(scope, gl::StrSpan(served)))
            preparedAtCase.push_back(std::string(payload.ptr, static_cast<std::size_t>(payload.len)));
    }
    bool completed = false;
    for (const std::string& payload : preparedAtCase) {
        ASSERT_TRUE(payload.rfind("(existence55[", 0) != 0);
        completed = completed || payload.rfind("(existence59[2,5,6,", 0) == 0;
    }
    ASSERT_TRUE(completed);
}

// A record owned at a scope is present there and at every scope below it,
// never above or beside it; an unregistered scope sees nothing, and the exact
// probe at a descendant does not see the owner's record.
TEST(ancestor_integration_demand, present_at_or_above_reads_the_scope_and_its_ancestors) {
    TrialFixture f;
    const std::string child = f.scope(gl::NameMap::MAIN_ID, "case_child");
    const gl::NameId childId = f.memory.nameMap.lookup(gl::StrSpan(child));
    const std::string grandchild = f.scope(childId, "case_grandchild");
    const std::string sibling = f.scope(gl::NameMap::MAIN_ID, "case_sibling");
    const std::string record = "(existence55[2,3,4,5,6])";
    f.memory.integrationPrepared.mint(gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap, record,
        child));
    gl::CanonicalTemplateMatcher& templates = f.analyzer.canonicalTemplates;
    const auto present = [&](const std::string& at) {
        return templates.canonicalTemplatePresentAtOrAbove(f.memory, f.memory.integrationPrepared,
            f.memory.lbMemory.canonicalUses.preparedMatches, gl::StrSpan(record), gl::StrSpan(at));
    };
    ASSERT_TRUE(present(child));
    ASSERT_TRUE(present(grandchild));
    ASSERT_FALSE(present("main"));
    ASSERT_FALSE(present(sibling));
    ASSERT_FALSE(present("missing_scope"));
    ASSERT_FALSE(templates.canonicalTemplatePresent(f.memory, f.memory.integrationPrepared,
        f.memory.lbMemory.canonicalUses.preparedMatches, gl::StrSpan(record), gl::StrSpan(grandchild)));
}
