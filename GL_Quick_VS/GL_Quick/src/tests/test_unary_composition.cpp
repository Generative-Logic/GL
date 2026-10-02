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
#include "../memory_infra/unary_composition.hpp"
#include "../prover.hpp"
#include <string>

namespace {
    int64_t addIntegrationDemand(gl::ExpressionAnalyzer& engine,const std::string& graph="13",
        const std::string& scope="main") {
        auto& memory=engine.body;
        const std::string marked="(in2[u_14,marker,u_"+graph+"])";
        const auto key=gl::mintTemplateKey(memory.templateInterner,memory.nameMap,marked,scope);
        gl::Instruction instruction;
        instruction.data.push_back(gl::LogicalEntity("atomic",{},marked,3));
        instruction.markedGoal=marked;
        gl::IntegrationEntry entry;
        entry.instruction=gl::encodeInstruction(instruction,memory.valueInterner);
        memory.overallHashMemory.admissionMapIntegration.assignRun(key,{entry});
        return key;
    }

    std::string installChain(gl::ExpressionAnalyzer& engine, const std::vector<std::string>& premises,
        const std::string& head, const std::string& scope = "main", const std::vector<int>& levels = {}) {
        const std::string proof=engine.reconstructImplicationFullBind(premises,head);
        auto& arena=gl::scratchArenas().forSlot(gl::scratchArenas().slotCount()-1);
        gl::ScratchScope frame(arena);
        gl::StrSpan chain[gl::ExecutionParameters::MAX_ADMISSION_KEY_ELEMENTS],
            remaining[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS], installedHead;
        int chainCount=0,remainingCount=0;
        engine.decomposeImplicationForInstall(gl::StrSpan(proof),arena,chain,chainCount,installedHead,remaining,remainingCount);
        engine.addToHashMemory(chain,chainCount,installedHead,remaining,remainingCount,engine.body,
            engine.body.overallHashMemory,levels.data(),static_cast<int32_t>(levels.size()),
            gl::StrSpan(proof),engine.parameters.maxIterationNumberVariable,
            engine.parameters.standardMaxSecondaryNumber,false,engine.parameters.minNumOperatorsKey,
            gl::StrSpan("implication",11),true,gl::StrSpan(proof),gl::StrSpan(scope));
        return proof;
    }

    std::string installUnary(gl::ExpressionAnalyzer& engine, const std::string& premise,
        const std::string& head, const std::string& scope = "main", const std::vector<int>& levels = {}) {
        return installChain(engine,{premise},head,scope,levels);
    }

    bool containsUnary(gl::ExpressionAnalyzer& engine, const std::string& premise,
        const std::string& head, const std::string& scope="main") {
        const auto& store=engine.body.lbMemory.unaryComposition;
        for(int32_t i=0;i<store.ruleInfo.size();++i) {
            const auto info=store.ruleInfo[i];
            if(engine.body.ruleInterner.decode(info.premise)==premise
                && engine.body.ruleInterner.decode(info.head)==head
                && engine.body.nameMap.decode(info.scope)==scope)return true;
        }
        return false;
    }

    std::pair<std::string, std::string> composition(
        const char* a, const char* b, const char* c, const char* d, bool expected = true) {
        gl::GlobalMemoryManager manager;
        manager.init(gl::StaticMemoryConfig{1 << 22, 1 << 18});
        gl::ScratchArena arena(&manager);
        gl::ScratchString premise, head;
        const auto match = gl::composeUnaryImplications(arena, gl::StrSpan(a), gl::StrSpan(b),
            gl::StrSpan(c), gl::StrSpan(d), premise, head);
        ASSERT_EQ(match == gl::UnaryCompositionMatch::matched, expected);
        if (!expected) return {};
        return {std::string(premise.data(), premise.size()), std::string(head.data(), head.size())};
    }
}

TEST(unary_composition, append_bridge_preserves_fixed_graphs) {
    const auto result = composition("(in2[i,j,u_f])", "(or8[i,j,u_f,u_a,u_k])",
        "(or8[x,y,u_f,u_a,u_k])", "(in2[x,y,u_g])");
    ASSERT_EQ(result.first, "(in2[1,2,u_f])");
    ASSERT_EQ(result.second, "(in2[1,2,u_g])");
}

TEST(unary_composition, complete_producer_chain_matching_preserves_every_premise) {
    gl::GlobalMemoryManager manager;
    manager.init(gl::StaticMemoryConfig{1 << 22, 1 << 18});
    gl::ScratchArena arena(&manager);
    const std::string first="(=[y,u_a])", second="!(P[x,y,u_n])";
    const gl::StrSpan premises[]={gl::StrSpan(first),gl::StrSpan(second)};
    gl::ScratchString chain,head;
    gl::normalizeCompositionChain(arena,premises,2,gl::StrSpan("(H[x,y,u_f])"),chain,head);
    ASSERT_EQ(std::string(chain.data(),chain.size()),"(=[1,u_a])\n!(P[2,1,u_n])");
    const auto joined=gl::joinCompositionPremises(arena,premises,2);
    ASSERT_EQ(std::string(joined.data(),joined.size()),first+"\n"+second);
    const auto match=gl::composeImplicationChain(arena,premises,2,gl::StrSpan("(H[x,y,u_f])"),
        gl::StrSpan("(H[q,p,u_f])"),gl::StrSpan("(G[q,p,u_g])"),chain,head);
    ASSERT_TRUE(match==gl::UnaryCompositionMatch::matched);
    ASSERT_EQ(std::string(chain.data(),chain.size()),"(=[1,u_a])\n!(P[2,1,u_n])");
    ASSERT_EQ(std::string(head.data(),head.size()),"(G[2,1,u_g])");
    for(const auto& incompatible : {"(H[q,q,u_f])","!(H[q,p,u_f])","(H[u_x,p,u_f])","(H[q,p,u_other])"}) {
        ASSERT_TRUE(gl::composeImplicationChain(arena,premises,2,gl::StrSpan("(H[x,y,u_f])"),
            gl::StrSpan(incompatible),gl::StrSpan("(G[q,p,u_g])"),chain,head)==gl::UnaryCompositionMatch::incompatible);
    }
    const gl::StrSpan extra[]={gl::StrSpan("(P[x,y,z])",10),gl::StrSpan(second)};
    ASSERT_TRUE(gl::composeImplicationChain(arena,extra,2,gl::StrSpan("(H[x,y,u_f])"),
        gl::StrSpan("(H[q,p,u_f])"),gl::StrSpan("(G[q,p,u_g])"),chain,head)==gl::UnaryCompositionMatch::incompatible);
}

TEST(unary_composition, multipremise_producer_requires_demand_in_all_arrival_orders) {
    std::vector<int> order={0,1,2};
    do {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        const std::vector<std::string> premises={"(=[1,u_12])","(=[2,u_10])"};
        const std::string expected="(=[1,u_12])\n(=[2,u_10])";
        bool demanded=false;
        for(const int step:order) {
            if(step==0)installChain(engine,premises,"(in2[2,1,u_11])");
            if(step==1)installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])");
            if(step==2){addIntegrationDemand(engine);demanded=true;}
            engine.drainUnaryComposition(engine.body);
            if(!demanded)ASSERT_FALSE(containsUnary(engine,expected,"(in2[2,1,u_13])"));
        }
        ASSERT_TRUE(containsUnary(engine,expected,"(in2[2,1,u_13])"));
        const int count=engine.body.lbMemory.unaryComposition.rules.count();
        engine.drainUnaryComposition(engine.body);
        ASSERT_EQ(engine.body.lbMemory.unaryComposition.rules.count(),count);
        for(int i=0;i<engine.body.lbMemory.unaryComposition.ruleInfo.size();++i) {
            const auto rule=engine.body.lbMemory.unaryComposition.ruleInfo[i];
            if(rule.premiseCount>1)ASSERT_EQ(engine.body.lbMemory.unaryComposition.consumerDemands.lookup(i+1),0);
        }
    }while(std::next_permutation(order.begin(),order.end()));
}

TEST(unary_composition, multipremise_consumer_and_sibling_sources_do_not_join) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    installUnary(engine,"(in2[1,2,u_10])","(in2[1,2,u_11])");
    installChain(engine,{"(in2[1,2,u_11])","(=[1,u_12])"},"(in2[1,2,u_13])");
    engine.drainUnaryComposition(engine.body);
    ASSERT_FALSE(containsUnary(engine,"(in2[1,2,u_10])","(in2[1,2,u_13])"));
    auto& names=engine.body.nameMap;
    const auto main=names.encode("main");
    const std::string left=names.decode(names.encodePush(main,"left"));
    const std::string right=names.decode(names.encodePush(main,"right"));
    installChain(engine,{"(=[1,u_12])","(=[2,u_10])"},"(in2[2,1,u_15])",left);
    installUnary(engine,"(in2[1,2,u_15])","(in2[1,2,u_13])",right);
    engine.drainUnaryComposition(engine.body);
    ASSERT_FALSE(containsUnary(engine,"(=[1,u_12])\n(=[2,u_10])","(in2[2,1,u_13])",right));
}

TEST(unary_composition, variables_are_standardized_apart) {
    const auto result = composition("(P[x,y])", "(Q[x,y])", "(Q[y,x])", "(R[x,y])");
    ASSERT_EQ(result.first, "(P[1,2])");
    ASSERT_EQ(result.second, "(R[2,1])");
}

TEST(unary_composition, repeated_intermediate_arguments_preserve_binders) {
    const auto result = composition("(P[x,y])", "(Q[x,y,x])", "(Q[a,b,a])", "(R[b,a])");
    ASSERT_EQ(result.first, "(P[1,2])");
    ASSERT_EQ(result.second, "(R[2,1])");
    composition("(P[x,y])", "(Q[x,y])", "(Q[z,z])", "(R[z])", false);
    composition("(P[x,y])", "(Q[x,x])", "(Q[a,b])", "(R[a,b])", false);
}

TEST(unary_composition, bound_variables_never_match_fixed_parameters) {
    composition("(P[x])", "(Q[x])", "(Q[u_a])", "(R[u_a])", false);
    composition("(P[u_a])", "(Q[u_a])", "(Q[x])", "(R[x])", false);
    composition("(=[1,u_10])", "(in[1,u_int_0_lev_4_10])",
        "(in[u_2,u_int_0_lev_4_10])", "(=[u_2,u_2])", false);
}

TEST(unary_composition, complete_binder_sets_must_match_and_survive) {
    composition("(P[x,y])", "(Q[x])", "(Q[a])", "(R[a])", false);
    composition("(P[x])", "(Q[x])", "(Q[a])", "(R[a,b])", false);
    composition("(P[x,y])", "(Q[x])", "(Q[a])", "(R[a,b])", false);
    composition("(P[x])", "(Q[x,y])", "(Q[a,b])", "(R[a])", false);
    const auto ground=composition("(P[u_a])", "(Q[u_a])", "(Q[u_a])", "(R[u_a])");
    ASSERT_EQ(ground.first,"(P[u_a])");
    ASSERT_EQ(ground.second,"(R[u_a])");
}

TEST(unary_composition, normalization_does_not_apply_the_composition_gate) {
    gl::GlobalMemoryManager manager;
    manager.init(gl::StaticMemoryConfig{1 << 22, 1 << 18});
    gl::ScratchArena arena(&manager);
    gl::ScratchString premise,head;
    gl::normalizeUnaryImplicationEndpoints(arena,gl::StrSpan("!(P[x,y,x,u_a])"),
        gl::StrSpan("(Q[y,z,u_a])"),premise,head);
    ASSERT_EQ(std::string(premise.data(),premise.size()),"!(P[1,2,1,u_a])");
    ASSERT_EQ(std::string(head.data(),head.size()),"(Q[2,3,u_a])");
    gl::normalizeUnaryImplicationEndpoints(arena,gl::StrSpan("(P[u_a])"),
        gl::StrSpan("(Q[u_a])"),premise,head);
    ASSERT_EQ(std::string(premise.data(),premise.size()),"(P[u_a])");
    ASSERT_EQ(std::string(head.data(),head.size()),"(Q[u_a])");
}

TEST(unary_composition, incompatible_signatures_and_fixed_parameters_do_not_match) {
    composition("(P[x])", "(Q[x])", "!(Q[y])", "(R[y])", false);
    composition("(P[x])", "(Q[x])", "(S[y])", "(R[y])", false);
    composition("(P[x])", "(Q[x])", "(Q[y,z])", "(R[y])", false);
    composition("(P[u_a])", "(Q[u_a])", "(Q[u_b])", "(R[u_b])", false);
    composition("(P[u_a,u_b])", "(Q[u_a,u_b])", "(Q[z,z])", "(R[z])", false);
}

TEST(unary_composition, negative_atoms_and_recursive_closure_are_canonical) {
    const auto first = composition("!(P[x])", "!(Q[x])", "!(Q[y])", "!(R[y])");
    ASSERT_EQ(first.first, "!(P[1])");
    ASSERT_EQ(first.second, "!(R[1])");
    const auto second = composition(first.first.c_str(), first.second.c_str(), "!(R[z])", "!(P[z])");
    ASSERT_EQ(second.first, "!(P[1])");
    ASSERT_EQ(second.second, "!(P[1])");
    const auto third = composition(second.first.c_str(), second.second.c_str(), "!(P[w])", "!(Q[w])");
    ASSERT_EQ(third.first, "!(P[1])");
    ASSERT_EQ(third.second, "!(Q[1])");
}

TEST(unary_composition, demanded_closure_stops_cycles) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    addIntegrationDemand(engine,"11");addIntegrationDemand(engine,"12");
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])");
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    installUnary(engine,"(in2[1,2,u_13])","(in2[1,2,u_11])");
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])"));
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_13])","(in2[1,2,u_12])"));
    const auto& store=engine.body.lbMemory.unaryComposition;
    ASSERT_EQ(store.rules.count(),6);
    ASSERT_TRUE(store.pending.empty());
    ASSERT_EQ(engine.body.overallHashMemory.admissionMapIntegration.count(),3);
    engine.drainUnaryComposition(engine.body);
    ASSERT_EQ(store.rules.count(),6);
}

TEST(unary_composition, matching_demand_is_required_in_either_arrival_order) {
    for(bool demandFirst:{false,true})for(bool reverse:{false,true}) {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        if(demandFirst)addIntegrationDemand(engine);
        installUnary(engine,reverse?"(in2[1,2,u_12])":"(in2[1,2,u_11])",
            reverse?"(in2[1,2,u_13])":"(in2[1,2,u_12])");
        installUnary(engine,reverse?"(in2[1,2,u_11])":"(in2[1,2,u_12])",
            reverse?"(in2[1,2,u_12])":"(in2[1,2,u_13])");
        engine.drainUnaryComposition(engine.body);
        if(!demandFirst){
            ASSERT_EQ(engine.body.lbMemory.unaryComposition.rules.count(),2);
            ASSERT_EQ(engine.body.lbMemory.unaryComposition.generated.count(),0);
            addIntegrationDemand(engine,"99");
            engine.drainUnaryComposition(engine.body);
            ASSERT_FALSE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])"));
            addIntegrationDemand(engine);
            engine.wakeUnaryCompositionDemands(engine.body);
            ASSERT_TRUE(!engine.body.lbMemory.unaryComposition.pending.empty());
            engine.drainUnaryComposition(engine.body);
        }
        ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])"));
        const int count=engine.body.lbMemory.unaryComposition.rules.count();
        engine.wakeUnaryCompositionDemands(engine.body);
        ASSERT_TRUE(engine.body.lbMemory.unaryComposition.pending.empty());
        engine.drainUnaryComposition(engine.body);
        ASSERT_EQ(engine.body.lbMemory.unaryComposition.rules.count(),count);
    }
}

TEST(unary_composition, cnfe_head_matching_promotes_only_the_witness) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    gl::StrReplacement mapping[gl::ExecutionParameters::MAX_ARITY];int32_t count=0;gl::StrSpan witness;
    const std::string headText="(in2[i,j,u_13])",demandText="(in2[u_14,marker,u_13])";
    ASSERT_TRUE(engine.matchUnaryAdmissionHead(gl::StrSpan(headText),
        gl::StrSpan(demandText),mapping,count,witness));
    ASSERT_EQ(std::string(witness.ptr,witness.len),"j");
    ASSERT_EQ(count,2);
    ASSERT_EQ(std::string(mapping[0].key.ptr,mapping[0].key.len),"i");
    ASSERT_EQ(std::string(mapping[0].value.ptr,mapping[0].value.len),"u_14");
    ASSERT_EQ(std::string(mapping[1].value.ptr,mapping[1].value.len),"marker");
    for(const auto& head:std::vector<std::string>{"(in2[i,j,u_12])","!(in2[i,j,u_13])",
        "(in2[i,u_j,u_13])","(in2[i,i,u_13])","(in2[i,j,k])","(in2[u_14,i,j])"})
        ASSERT_FALSE(engine.matchUnaryAdmissionHead(gl::StrSpan(head),
            gl::StrSpan("(in2[u_14,marker,u_13])"),mapping,count,witness));
    for(const auto& demand:std::vector<std::string>{"(in2[u_14,u_marker,u_13])",
        "(in2[marker,marker,u_13])","(in[u_14,marker])"})
        ASSERT_FALSE(engine.matchUnaryAdmissionHead(gl::StrSpan("(in2[i,j,u_13])"),
            gl::StrSpan(demand),mapping,count,witness));
}

TEST(unary_composition, consumer_index_rejects_wrong_demands_and_incompatible_scopes) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    const auto wrong=addIntegrationDemand(engine,"99");
    ASSERT_FALSE(engine.indexUnaryCompositionDemand(engine.body,1,wrong));
    const auto demand=addIntegrationDemand(engine);
    ASSERT_TRUE(engine.indexUnaryCompositionDemand(engine.body,1,demand));
    ASSERT_TRUE(engine.indexUnaryCompositionDemand(engine.body,1,demand));
    const auto& state=engine.body.lbMemory.unaryComposition;
    ASSERT_EQ(state.consumerDemands.runLen(state.consumerDemands.lookup(1)),1);
    auto& names=engine.body.nameMap;
    const auto left=names.encodePush(gl::NameMap::MAIN_ID,"left");
    const auto right=names.encodePush(gl::NameMap::MAIN_ID,"right");
    const std::string leftScope=names.decode(left),rightScope=names.decode(right);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])",rightScope);
    const auto scoped=addIntegrationDemand(engine,"12",leftScope);
    ASSERT_FALSE(engine.indexUnaryCompositionDemand(engine.body,2,scoped));
}

TEST(unary_composition, demand_scope_must_be_comparable_to_the_composed_scope) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    auto& names=engine.body.nameMap;
    const std::string left=names.decode(names.encodePush(gl::NameMap::MAIN_ID,"left"));
    const std::string right=names.decode(names.encodePush(gl::NameMap::MAIN_ID,"right"));
    addIntegrationDemand(engine,"13",left);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])",right);
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    engine.drainUnaryComposition(engine.body);
    ASSERT_FALSE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])",right));
    addIntegrationDemand(engine,"13",right);
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])",right));
}

TEST(unary_composition, removed_and_reintroduced_demand_wakes_new_producers) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    engine.drainUnaryComposition(engine.body);
    engine.body.overallHashMemory.admissionMapIntegration.resetToFresh();
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(engine.body.lbMemory.unaryComposition.observedDemands.empty());
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])");
    engine.drainUnaryComposition(engine.body);
    ASSERT_FALSE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])"));
    addIntegrationDemand(engine);
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])"));
}

TEST(unary_composition, either_rule_arrival_order_produces_the_composition) {
    for(bool reverse:{false,true}) {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        addIntegrationDemand(engine);
        installUnary(engine,reverse ? "(in2[1,2,u_12])":"(in2[1,2,u_11])",
            reverse ? "(in2[1,2,u_13])":"(in2[1,2,u_12])");
        engine.drainUnaryComposition(engine.body);
        installUnary(engine,reverse ? "(in2[1,2,u_11])":"(in2[1,2,u_12])",
            reverse ? "(in2[1,2,u_12])":"(in2[1,2,u_13])");
        engine.drainUnaryComposition(engine.body);
        ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])"));
    }
}

TEST(unary_composition, closure_flushes_all_generated_rules_before_readers_resume) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine,"31");
    for(int i=0;i<20;++i)
        installUnary(engine,"(in2[1,2,u_"+std::to_string(11+i)+"])",
            "(in2[1,2,u_"+std::to_string(12+i)+"])");
    engine.drainUnaryComposition(engine.body);
    ASSERT_EQ(engine.body.ruleStagingArmDepth,0);
    ASSERT_TRUE(gl::ruleStagings().slotIsEmpty(engine.currentGenSlot()));
    for(int i=0;i<20;++i)
        ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_"+std::to_string(11+i)+"])",
            "(in2[1,2,u_"+std::to_string(31)+"])") );
    ASSERT_EQ(engine.body.overallHashMemory.originals.count(),39);
    ASSERT_EQ(engine.body.localHashMemory.originals.count(),19);
    ASSERT_EQ(engine.body.localHashMemoryDelta.originals.count(),19);
    ASSERT_TRUE(engine.body.lbMemory.unaryComposition.pending.empty());
    engine.drainUnaryComposition(engine.body);
    ASSERT_EQ(engine.body.overallHashMemory.originals.count(),39);
    ASSERT_EQ(engine.body.ruleStagingArmDepth,0);
    ASSERT_TRUE(gl::ruleStagings().slotIsEmpty(engine.currentGenSlot()));
}

TEST(unary_composition, joins_use_deepest_comparable_scope) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    addIntegrationDemand(engine,"14");
    auto& names=engine.body.nameMap;
    const auto main=names.encode("main");
    const auto left=names.encodePush(main,"left");
    const auto right=names.encodePush(main,"right");
    const std::string leftName=names.decode(left),rightName=names.decode(right);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])","main",{1});
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])",leftName,{2});
    installUnary(engine,"(in2[1,2,u_13])","(in2[1,2,u_14])",rightName,{3});
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])",leftName));
    ASSERT_FALSE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])","main"));
    ASSERT_FALSE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_14])",rightName));
}

TEST(unary_composition, removal_rebuilds_from_roots_without_cyclic_support) {
    for (const bool multiple : {false,true}) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    addIntegrationDemand(engine,"11");addIntegrationDemand(engine,"12");
    std::vector<std::string> premises={"(in2[1,2,u_11])"};
    if(multiple)premises.push_back("(=[1,u_20])");
    const std::string expected=premises[0]+(multiple?"\n"+premises[1]:"");
    const std::string first=installChain(engine,premises,"(in2[1,2,u_12])");
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    installUnary(engine,"(in2[1,2,u_13])","(in2[1,2,u_11])");
    engine.drainUnaryComposition(engine.body);
    gl::LbArena arena(&gl::staticMemory());
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::ColdHashSet<gl::BytesKeyStore> visited(&arena,&dirty);
    gl::ColdHashSet<gl::PodKeyStore<int64_t>> dropped(&arena,&dirty);
    engine.removeRuleFromHashMemory(engine.body,engine.body.overallHashMemory,
        gl::StrSpan(first),gl::StrSpan("main",4),visited,dropped);
    engine.finishRuleRemovals(engine.body.overallHashMemory,dropped,arena);
    engine.drainUnaryComposition(engine.body);
    ASSERT_FALSE(containsUnary(engine,expected,"(in2[1,2,u_13])"));
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_11])"));
    }
}

TEST(unary_composition, admission_transfer_preserves_instruction_and_both_arrival_orders) {
    for(bool demandFirst:{false,true}) {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        engine.parameters.ban_disintegration=false;
        auto& memory=engine.body;
        const std::string marked="(in2[u_14,marker,u_13])";
        const auto key=gl::mintTemplateKey(memory.templateInterner,memory.nameMap,marked,"main");
        gl::Instruction instruction;
        instruction.data.push_back(gl::LogicalEntity("atomic",{},marked,3));
        instruction.markedGoal=marked;
        gl::IntegrationEntry entry;
        entry.instruction=gl::encodeInstruction(instruction,memory.valueInterner);
        const auto before=gl::Codec<gl::IntegrationEntry>::serialize(entry);
        if(demandFirst)memory.overallHashMemory.admissionMapIntegration.assignRun(key,{entry});
        engine.transferUnaryIntegrationDemands(memory);
        installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])");
        engine.drainUnaryComposition(memory);
        engine.transferUnaryIntegrationDemands(memory);
        if(!demandFirst)memory.overallHashMemory.admissionMapIntegration.assignRun(key,{entry});
        engine.transferUnaryIntegrationDemands(memory);
        int64_t algebraKey=0;
        ASSERT_TRUE(gl::lookupTemplateKey(memory.templateInterner,memory.nameMap,
            gl::StrSpan("(in2[14,marker,11])"),gl::StrSpan("main",4),algebraKey));
        ASSERT_TRUE(algebraKey!=0);
        ASSERT_TRUE(memory.overallHashMemory.admissionMap.lookup(algebraKey)!=0);
        const int row=memory.overallHashMemory.admissionMapIntegration.lookup(key);
        ASSERT_TRUE(before==gl::Codec<gl::IntegrationEntry>::serialize(
            memory.overallHashMemory.admissionMapIntegration.recordAt(row,0)));
        const int demands=memory.overallHashMemory.admissionMap.count();
        engine.transferUnaryIntegrationDemands(memory);
        ASSERT_EQ(memory.overallHashMemory.admissionMap.count(),demands);
        const gl::WitnessAdmissionMetrics fits{engine.parameters.maxIterationNumberVariable,1};
        const gl::WitnessAdmissionMetrics tooDeep{engine.parameters.maxIterationNumberVariable+1,1};
        ASSERT_TRUE(engine.isAdmittedIncludingAncestors(memory,gl::StrSpan("(in2[14,w,11])"),
            gl::StrSpan("w"),gl::StrSpan("(in2[14,marker,11])"),gl::StrSpan("main"),&fits,false));
        ASSERT_FALSE(engine.isAdmittedIncludingAncestors(memory,gl::StrSpan("(in2[14,w,11])"),
            gl::StrSpan("w"),gl::StrSpan("(in2[14,marker,11])"),gl::StrSpan("main"),&tooDeep,false));
    }
}

TEST(unary_composition, admission_transfer_walk_skips_only_unchanged_key_sets) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    engine.parameters.ban_disintegration=false;
    auto& memory=engine.body;
    auto& store=memory.lbMemory.unaryComposition;
    const auto transferred=[&](const char* text) {
        int64_t key=0;
        return gl::lookupTemplateKey(memory.templateInterner,memory.nameMap,
                gl::StrSpan(text),gl::StrSpan("main",4),key)
            && key!=0 && memory.overallHashMemory.admissionMap.lookup(key)!=0;
    };
    const std::string first=installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])");
    engine.drainUnaryComposition(memory);
    engine.transferUnaryIntegrationDemands(memory);
    ASSERT_EQ(store.admissionTransfers.count(),0);
    // A demand arriving after a walk is transferred by the next call.
    addIntegrationDemand(engine);
    engine.transferUnaryIntegrationDemands(memory);
    ASSERT_TRUE(transferred("(in2[14,marker,11])"));
    const int32_t receipts=store.admissionTransfers.count();
    ASSERT_TRUE(receipts>0);
    // Unchanged key sets: the recorded epochs stand and nothing changes.
    const int32_t demands=memory.overallHashMemory.admissionMap.count();
    engine.transferUnaryIntegrationDemands(memory);
    ASSERT_EQ(store.admissionTransfers.count(),receipts);
    ASSERT_EQ(memory.overallHashMemory.admissionMap.count(),demands);
    ASSERT_TRUE(store.transferWalked);
    ASSERT_EQ(store.transferDemandEpoch,
        memory.overallHashMemory.admissionMapIntegration.inner().insertEpoch());
    ASSERT_EQ(store.transferRuleEpoch,store.rules.insertEpoch());
    // A rule arriving after a walk meets the standing demand.
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    engine.drainUnaryComposition(memory);
    ASSERT_FALSE(transferred("(in2[14,marker,12])"));
    engine.transferUnaryIntegrationDemands(memory);
    ASSERT_TRUE(transferred("(in2[14,marker,12])"));
    // A closure rebuild clears the receipts; the next call walks again.
    engine.forgetUnaryImplication(memory,gl::StrSpan(first),gl::StrSpan("main"));
    engine.drainUnaryComposition(memory);
    ASSERT_EQ(store.admissionTransfers.count(),0);
    engine.transferUnaryIntegrationDemands(memory);
    ASSERT_TRUE(store.admissionTransfers.count()>0);
}

TEST(unary_composition, independent_support_survives_generated_owner_rebuild) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    const std::string first=installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])");
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    engine.drainUnaryComposition(engine.body);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])");
    engine.forgetUnaryImplication(engine.body,gl::StrSpan(first),gl::StrSpan("main"));
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])"));
    ASSERT_TRUE(engine.body.overallHashMemory.originals.count()>0);
}

TEST(unary_composition, supporting_level_variants_remain_distinct) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])","main",{3});
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])","main",{5});
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])","main",{7});
    engine.drainUnaryComposition(engine.body);
    std::set<std::vector<int32_t>> supports;
    const auto& store=engine.body.lbMemory.unaryComposition;
    for(int i=0;i<store.ruleInfo.size();++i) {
        const auto rule=store.ruleInfo[i];
        if(engine.body.ruleInterner.decode(rule.premise)!="(in2[1,2,u_11])"
            || engine.body.ruleInterner.decode(rule.head)!="(in2[1,2,u_13])")continue;
        const auto key=store.rules.keyAt(i+1);
        int32_t count;std::memcpy(&count,key.ptr+12,4);
        std::vector<int32_t> levels(count);std::memcpy(levels.data(),key.ptr+16,count*4);
        supports.insert(levels);
    }
    ASSERT_TRUE((supports==std::set<std::vector<int32_t>>{{3,5},{3,7}}));
}

TEST(unary_composition, scope_cleanup_retires_only_scoped_support) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    auto& names=engine.body.nameMap;
    const auto child=names.encodePush(names.encode("main"),"composition_child");
    const std::string scope=names.decode(child);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])");
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])",scope);
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])",scope));
    engine.body.wipeSubtree(child);
    engine.drainUnaryComposition(engine.body);
    ASSERT_FALSE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])",scope));
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])"));
}

TEST(unary_composition, admission_payload_keeps_numeric_placeholder_and_fixed_arguments) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    engine.parameters.ban_disintegration=false;
    auto& memory=engine.body;
    const int level=0;
    engine.addExprToMemoryBlock(gl::StrSpan("(=[21,22])"),memory,0,1,&level,1,
        gl::TransientOrigin{true,gl::OriginTag::taskFormulation,nullptr,0},-1,-1,gl::StrSpan("main"),false);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])");
    const std::string marked="(in2[u_repl_lev_4_4,marker,u_13])";
    const auto key=gl::mintTemplateKey(memory.templateInterner,memory.nameMap,marked,"main");
    gl::Instruction instruction;
    instruction.data.push_back(gl::LogicalEntity("atomic",{},marked,3));instruction.markedGoal=marked;
    gl::IntegrationEntry entry;entry.instruction=gl::encodeInstruction(instruction,memory.valueInterner);
    memory.overallHashMemory.admissionMapIntegration.assignRun(key,{entry});
    engine.transferUnaryIntegrationDemands(memory);
    int64_t algebra=0;
    ASSERT_TRUE(gl::lookupTemplateKey(memory.templateInterner,memory.nameMap,
        gl::StrSpan("(in2[repl_lev_4_4,marker,11])"),gl::StrSpan("main"),algebra));
    const auto value=memory.overallHashMemory.admissionMap.recordAt(
        memory.overallHashMemory.admissionMap.lookup(algebra),0);
    ASSERT_EQ(value.key.size(),size_t(1));
    ASSERT_TRUE(memory.valueInterner.decode(value.key[0]).find("marker")==std::string::npos);
    std::set<std::string> fixed;
    for(auto id:value.remainingArgs)fixed.insert(memory.valueInterner.decode(id));
    ASSERT_TRUE((fixed==std::set<std::string>{"11","repl_lev_4_4"}));
    ASSERT_TRUE(engine.isAdmitted(memory,gl::StrSpan("(in2[repl_lev_4_4,it_0_lev_0_1,11])"),
        gl::StrSpan("it_0_lev_0_1"),gl::StrSpan("(in2[repl_lev_4_4,marker,11])"),gl::StrSpan("main")));
}

TEST(unary_composition, shared_premise_composition_keeps_request_keys) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])");
    installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    auto& hash=engine.body.overallHashMemory;
    const int before=hash.normalizedEncodedKeys.count();
    const int statements=engine.body.intEncodedStatements.size();
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])"));
    ASSERT_EQ(hash.normalizedEncodedKeys.count(),before);
    ASSERT_EQ(engine.body.intEncodedStatements.size(),statements);
}

TEST(unary_composition, admission_transfer_rejects_invalid_patterns) {
    const std::vector<std::string> heads={
        "(in2[1,2,u_12])", "!(in2[1,2,u_13])", "(in2[1,u_w,u_13])",
        "(in2[1,1,u_13])", "(in2[1,2,3])", "(in2[u_14,1,2])"};
    for(const auto& head:heads) {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        engine.parameters.ban_disintegration=false;
        auto& memory=engine.body;
        installUnary(engine,head=="(in2[u_14,1,2])" ? "(in2[1,u_w,2])" : "(in2[1,2,u_11])",head);
        const std::string marked="(in2[u_14,marker,u_13])";
        const auto key=gl::mintTemplateKey(memory.templateInterner,memory.nameMap,marked,"main");
        gl::Instruction instruction;
        instruction.data.push_back(gl::LogicalEntity("atomic",{},marked,3));instruction.markedGoal=marked;
        gl::IntegrationEntry entry;entry.instruction=gl::encodeInstruction(instruction,memory.valueInterner);
        memory.overallHashMemory.admissionMapIntegration.assignRun(key,{entry});
        const int countBefore=memory.overallHashMemory.admissionMap.count();
        engine.transferUnaryIntegrationDemands(memory);
        ASSERT_EQ(memory.overallHashMemory.admissionMap.count(),countBefore);
        int64_t algebra=0;
        const bool found=gl::lookupTemplateKey(memory.templateInterner,memory.nameMap,
            gl::StrSpan("(in2[14,marker,11])"),gl::StrSpan("main"),algebra);
        ASSERT_TRUE(!found || memory.overallHashMemory.admissionMap.lookup(algebra)==0);
    }
}

TEST(unary_composition, composed_rule_history_ships_without_a_statement_delta) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine);
    engine.parameters.trackHistory=true;
    const auto first=installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_12])");
    const auto second=installUnary(engine,"(in2[1,2,u_12])","(in2[1,2,u_13])");
    const int before=engine.body.intLocalEncodedStatementsDelta.size();
    engine.drainUnaryComposition(engine.body);
    ASSERT_EQ(engine.body.intLocalEncodedStatementsDelta.size(),before);
    const auto rows=gl::decodeMailOutOrigins(engine.body.mailOut,engine.body.mailOutInterner);
    bool found=false;
    for(const auto& row:rows) {
        if(row.first.original!="(>[1,2](in2[1,2,u_11])(in2[1,2,u_13]))")continue;
        ASSERT_EQ(row.first.validityName,std::string("main"));
        for(const auto& origin:row.second) {
            if(origin.first!="implication composition")continue;
            ASSERT_EQ(origin.second.size(),size_t(2));
            ASSERT_EQ(origin.second[0].original,first);
            ASSERT_EQ(origin.second[1].original,second);
            ASSERT_EQ(origin.second[0].validityName,std::string("main"));
            ASSERT_EQ(origin.second[1].validityName,std::string("main"));
            found=true;
        }
    }
    ASSERT_TRUE(found);
}

TEST(unary_composition, canonical_source_reinstallation_retires_old_derived_rules) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    addIntegrationDemand(engine,"c");
    const auto original=installUnary(engine,"(in2[1,2,u_z])","(in2[1,2,u_b])");
    installUnary(engine,"(in2[1,2,u_b])","(in2[1,2,u_c])");
    engine.drainUnaryComposition(engine.body);
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_z])","(in2[1,2,u_c])"));
    gl::LbArena arena(&gl::staticMemory());
    gl::DirtyState dirty=gl::DirtyState::Clean;
    gl::ColdHashSet<gl::BytesKeyStore> visited(&arena,&dirty);
    gl::ColdHashSet<gl::PodKeyStore<int64_t>> dropped(&arena,&dirty);
    engine.removeRuleFromHashMemory(engine.body,engine.body.overallHashMemory,
        gl::StrSpan(original),gl::StrSpan("main"),visited,dropped);
    engine.finishRuleRemovals(engine.body.overallHashMemory,dropped,arena);
    installUnary(engine,"(in2[1,2,u_a])","(in2[1,2,u_b])");
    engine.drainUnaryComposition(engine.body);
    ASSERT_FALSE(containsUnary(engine,"(in2[1,2,u_z])","(in2[1,2,u_c])"));
    ASSERT_TRUE(containsUnary(engine,"(in2[1,2,u_a])","(in2[1,2,u_c])"));
}

TEST(unary_composition, admission_transfer_rejects_sibling_scopes) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    engine.parameters.ban_disintegration=false;
    auto& memory=engine.body;
    const auto left=memory.nameMap.encodePush(gl::NameMap::MAIN_ID,"left");
    const auto right=memory.nameMap.encodePush(gl::NameMap::MAIN_ID,"right");
    const std::string leftScope=memory.nameMap.decode(left),rightScope=memory.nameMap.decode(right);
    installUnary(engine,"(in2[1,2,u_11])","(in2[1,2,u_13])",leftScope);
    const std::string marked="(in2[u_14,marker,u_13])";
    const auto key=gl::mintTemplateKey(memory.templateInterner,memory.nameMap,marked,rightScope);
    gl::Instruction instruction;
    instruction.data.push_back(gl::LogicalEntity("atomic",{},marked,3));instruction.markedGoal=marked;
    gl::IntegrationEntry entry;entry.instruction=gl::encodeInstruction(instruction,memory.valueInterner);
    memory.overallHashMemory.admissionMapIntegration.assignRun(key,{entry});
    const int before=memory.overallHashMemory.admissionMap.count();
    engine.transferUnaryIntegrationDemands(memory);
    ASSERT_EQ(memory.overallHashMemory.admissionMap.count(),before);
}

TEST(unary_composition, demand_gates_equality_copy_and_preserves_instruction) {
    for(bool demandFirst:{false,true})for(bool reversed:{false,true}) {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        engine.parameters.ban_disintegration=false;
        engine.parameters.trackHistory=true;
        auto& memory=engine.body;
        int64_t key=0;
        if(demandFirst)key=addIntegrationDemand(engine);
        engine.transferUnaryIntegrationDemands(memory);
        ASSERT_TRUE(memory.sameIterationInternalMail.statements_.empty());
        installChain(engine,{reversed?"(=[u_12,1])":"(=[1,u_12])","(=[2,u_10])"},"(in2[2,1,u_13])");
        if(!demandFirst) {
            engine.transferUnaryIntegrationDemands(memory);
            ASSERT_TRUE(memory.sameIterationInternalMail.statements_.empty());
            key=addIntegrationDemand(engine);
        }
        const auto before=memory.overallHashMemory.admissionMapIntegration.recordsAt(
            memory.overallHashMemory.admissionMapIntegration.lookup(key));
        engine.transferUnaryIntegrationDemands(memory);
        ASSERT_EQ(memory.sameIterationInternalMail.statements_.count(),1);
        const auto copies=gl::decodeInternalMailStatements(memory.sameIterationInternalMail,memory.nameMap);
        ASSERT_EQ(copies[0].first.original,std::string("(=[12,12_copy])"));
        ASSERT_EQ(copies[0].first.validityName,std::string("main"));
        const auto origins=gl::makeHeapMail(memory.sameIterationInternalMail,memory.nameMap,memory.originInterner);
        ASSERT_EQ(origins.exprOriginMap.size(),size_t(1));
        ASSERT_EQ(origins.exprOriginMap.begin()->second.begin()->first,std::string("variable copy"));
        ASSERT_TRUE(origins.exprOriginMap.begin()->second.begin()->second.empty());
        engine.transferUnaryIntegrationDemands(memory);
        ASSERT_EQ(memory.sameIterationInternalMail.statements_.count(),1);
        const auto after=memory.overallHashMemory.admissionMapIntegration.recordsAt(
            memory.overallHashMemory.admissionMapIntegration.lookup(key));
        ASSERT_EQ(before.size(),after.size());
        ASSERT_TRUE(gl::Codec<gl::IntegrationEntry>::serialize(before[0])==gl::Codec<gl::IntegrationEntry>::serialize(after[0]));
    }
}

TEST(unary_composition, copy_requires_positive_marker_equality_and_exact_head_demand) {
    for(const auto& equality:{"!(=[1,u_12])","(=[2,u_10])","(=[u_12,u_10])","(=[3,u_12])","(=[1,3])"}) {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        engine.parameters.ban_disintegration=false;
        addIntegrationDemand(engine);
        installChain(engine,{equality,"(in2[2,1,u_11])"},"(in2[2,1,u_13])");
        engine.transferUnaryIntegrationDemands(engine.body);
        ASSERT_TRUE(engine.body.sameIterationInternalMail.statements_.empty());
    }
    for(const auto& head:{"!(in2[2,1,u_13])","(in2[2,1,u_15])","(in2[2,2,u_13])"}) {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        engine.parameters.ban_disintegration=false;
        addIntegrationDemand(engine);
        installChain(engine,{"(=[1,u_12])","(=[2,u_10])"},head);
        engine.transferUnaryIntegrationDemands(engine.body);
        ASSERT_TRUE(engine.body.sameIterationInternalMail.statements_.empty());
    }
}

TEST(unary_composition, equality_copy_reuses_visible_alias_and_respects_scope) {
    for(bool sibling:{false,true})for(bool existing:{false,true}) {
        gl::ExpressionAnalyzer engine(std::string("Peano"));
        engine.parameters.ban_disintegration=false;
        auto& memory=engine.body;
        const auto left=memory.nameMap.encodePush(gl::NameMap::MAIN_ID,"left");
        const auto target=memory.nameMap.encodePush(sibling?gl::NameMap::MAIN_ID:left,"right");
        const std::string sourceScope=memory.nameMap.decode(left),targetScope=memory.nameMap.decode(target);
        if(existing) {
            const int levels[]={0};
            engine.addExprToMemoryBlock(gl::StrSpan("(=[12,15])",10),memory,0,0,levels,1,
                gl::TransientOrigin{true,gl::OriginTag::taskFormulation,nullptr,0},-1,-1,gl::StrSpan(sourceScope),false);
        }
        addIntegrationDemand(engine,"13",targetScope);
        installChain(engine,{"(=[1,u_12])","(=[2,u_10])"},"(in2[2,1,u_13])",sourceScope);
        engine.transferUnaryIntegrationDemands(memory);
        ASSERT_EQ(memory.sameIterationInternalMail.statements_.count(),(!sibling && !existing)?1:0);
        if(!sibling && !existing) {
            const auto copies=gl::decodeInternalMailStatements(memory.sameIterationInternalMail,memory.nameMap);
            ASSERT_EQ(copies[0].first.validityName,targetScope);
        }
    }
}

TEST(unary_composition, copied_value_fires_only_with_the_other_endpoint_premise) {
    gl::ExpressionAnalyzer engine(std::string("Peano"));
    engine.parameters.ban_disintegration=false;
    auto& memory=engine.body;
    addIntegrationDemand(engine);
    installChain(engine,{"(=[1,u_12])","(=[2,u_10])"},"(in2[2,1,u_13])");
    engine.transferUnaryIntegrationDemands(memory);
    engine.standardProcessing(memory,nullptr,memory.sameIterationInternalMail,memory.nextIterationInternalMail,-1);
    ASSERT_TRUE(gl::lookupStatementFlags(memory.intKnownStatements,memory.nameMap,
        gl::StrSpan("(=[12_copy,12])",15),gl::StrSpan("main",4))!=nullptr);
    for(bool endpoint:{false,true}) {
        if(endpoint) {
            const int levels[]={0};
            engine.addExprToMemoryBlock(gl::StrSpan("(=[14,10])",10),memory,0,0,levels,1,
                gl::TransientOrigin{true,gl::OriginTag::taskFormulation,nullptr,0},-1,-1,gl::StrSpan("main",4),false);
        }
        gl::SealedPageSet pages;
        pages.bind(&gl::staticMemory());
        std::atomic<int64_t> doomLine{gl::kNoDoomLine};
        gl::BurstSink sink{&engine,&memory,0u,&pages,&doomLine,0,gl::SealedRecordCursor<gl::FiringRecord>(pages)};
        engine.generateEncodedRequestsStatic(memory,memory.overallHashMemory,nullptr,0,gl::SplitStumpRef{},0u,sink);
        pages.seal();
        bool fired=false;
        pages.forEachRecord<gl::FiringRecord>([&](const gl::FiringRecord& record) {
            if(!record.isMarker && !record.isOrdis2Demand && record.rplExpr2.toStdString()=="(in2[14,12_copy,13])") {
                ASSERT_EQ(record.validityName.toStdString(),std::string("main"));
                ASSERT_EQ(record.originTag.toStdString(),std::string("implication"));
                fired=true;
            }
        });
        pages.freePages();
        ASSERT_EQ(fired,endpoint);
    }
}
