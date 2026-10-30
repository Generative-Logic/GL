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
/// @brief Unit tests for demand-built rules (D-596,
///        D-597): the shape criterion, the witness
///        and trigger matches, the instantiation, the process-wide registry,
///        the install decision and the demand-side build of
///        `RejectedSourceReplay`.
///
/// @details
/// The Peano fixture supplies the compiled definitions: `existence1` is
/// `∃ z ∈ u_1: in3[u_2,u_3,z,u_4]`, `existence0` the unary
/// `∃ z ∈ u_1: in2[u_2,z,u_3]`; the OR heads are compiled in the tests from
/// `preorder` disjuncts. Every test that registers a rule clears the
/// process-wide registry before and after, through its fixture.

#include "../logical_block/demand_built_rules.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "demand_built_rule_registry_fixture.hpp"
#include "../prover.hpp"
#include "../logical_block/statement_arrivals.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace {

gl::StrSpan sp(const char* text) {
    return gl::StrSpan(text, static_cast<int32_t>(std::strlen(text)));
}

const std::string kSumRule = "(>[1](in[1,u_1])(>[2](in[2,u_1])(existence1[u_1,1,2,u_4])))";
const std::string kSumCarrier = "(implication18[1,1,1,4])";

/// The Peano anchor's arguments, bare.
struct PeanoAnchors {
    gl::StrSpan spans[6] = { gl::StrSpan("1", 1), gl::StrSpan("2", 1), gl::StrSpan("3", 1),
                             gl::StrSpan("4", 1), gl::StrSpan("5", 1), gl::StrSpan("6", 1) };
};

/// Clears the process-wide registry on entry and exit.
struct RegistryGuard {
    RegistryGuard() { gl::tests::DemandBuiltRuleRegistryFixture::clear(); }
    ~RegistryGuard() { gl::tests::DemandBuiltRuleRegistryFixture::clear(); }
};

bool analyze(const gl::CompiledDefinitionRegistry& definitions, const std::string& rule,
             const gl::StrSpan* anchors, int32_t anchorN, gl::DemandBuiltRule& out) {
    return gl::analyzeDemandBuiltRule(gl::StrSpan(rule), gl::StrSpan(kSumCarrier), definitions,
                                      gl::StrSpan("AnchorPeano", 11), anchors, anchorN, out);
}

/// A typed theorem rule of the Peano anchor over 9 and 10 with @p head.
std::string typedOrRule(const gl::tests::ProverFixture& ea, const std::string& head) {
    return "(>[1,2,3,4,5,6](" + ea.anchorInfo.name + "[1,2,3,4,5,6])(>[9](in[9,1])(>[10](in[10,1])"
        + head + ")))";
}

/// The templates of @p rule for one trigger kind, in template order.
std::vector<std::string> templatesOf(const gl::DemandBuiltRule& rule, gl::DemandBuiltTrigger kind) {
    std::vector<std::string> templates;
    for (int32_t t = 0; t < rule.triggerCount; ++t)
        if (rule.triggerKinds[t] == kind) templates.push_back(rule.triggerSpan(t).toStdString());
    return templates;
}

/// Whether @p list holds @p text.
bool holds(const std::vector<std::string>& list, const std::string& text) {
    return std::find(list.begin(), list.end(), text) != list.end();
}

/// A block on the anchor's own key, knowing (in[a,1]), (in[b,1]) and the
/// carrier at main; @p holdRule records the sum rule in its installed-rule
/// index.
void prepareBlock(gl::tests::ProverFixture& ea, gl::Memory& mb, bool holdRule) {
    mb.setExprKey("(" + ea.anchorInfo.name + "[1,2,3,4,5,6])");
    const gl::NameId main = gl::NameMap::MAIN_ID;
    const int level = 0;
    for (const char* known : { "(in[a,1])", "(in[b,1])", "(implication18[1,1,1,4])" }) {
        const gl::NameId id = mb.nameMap.encode(known);
        mb.intKnownStatements.insert(gl::StatementKey{ id, main }, gl::StatementFlags{ true, false });
        mb.intStatementLevelsMap.assignSet(gl::StatementKey{ id, main }, &level, 1);
    }
    if (holdRule)
        mb.expandedImplications.mint(gl::LbStatePairKey{ mb.lbStateInterner.encode(kSumRule),
                                                         mb.lbStateInterner.encode("main") });
}

}

TEST(demand_built_rules, analyze_accepts_an_anchor_operator_totality_rule) {
    gl::tests::ProverFixture ea("Peano");
    const PeanoAnchors anchors;
    gl::DemandBuiltRule rule;
    ASSERT_TRUE(analyze(ea.definitions, kSumRule, anchors.spans, 6, rule));
    ASSERT_EQ(rule.premiseCount, 2);
    ASSERT_TRUE(gl::equalSpans(rule.witnessSpan(), sp("(in3[1,2,marker,u_4])")));
    ASSERT_TRUE(gl::equalSpans(rule.sliceSpan(rule.premises[0]), sp("(in[1,u_1])")));
    ASSERT_TRUE(gl::equalSpans(rule.sliceSpan(rule.bound[1]), sp("2")));
    ASSERT_TRUE(gl::equalSpans(rule.sliceSpan(rule.head), sp("(existence1[u_1,1,2,u_4])")));
    ASSERT_TRUE(gl::equalSpans(rule.carrierSpan(), gl::StrSpan(kSumCarrier)));
}

TEST(demand_built_rules, analyze_rejects_every_other_shape) {
    gl::tests::ProverFixture ea("Peano");
    const PeanoAnchors anchors;
    gl::DemandBuiltRule rule;
    // The set is not an anchor argument.
    ASSERT_FALSE(analyze(ea.definitions, kSumRule, anchors.spans + 1, 5, rule));
    // The operator is not an anchor argument.
    const gl::StrSpan withoutPlus[5] = { anchors.spans[0], anchors.spans[1], anchors.spans[2],
                                         anchors.spans[4], anchors.spans[5] };
    ASSERT_FALSE(analyze(ea.definitions, kSumRule, withoutPlus, 5, rule));
    // No anchor at all.
    ASSERT_FALSE(analyze(ea.definitions, kSumRule, anchors.spans, 0, rule));
    // A premise other than a typing.
    ASSERT_FALSE(analyze(ea.definitions,
        "(>[1](in[1,u_1])(>[2](in2[2,u_1,u_3])(existence1[u_1,1,2,u_4])))", anchors.spans, 6, rule));
    // One typed variable: the unary operator's totality stays eager.
    ASSERT_FALSE(analyze(ea.definitions, "(>[1](in[1,u_1])(existence0[u_1,1,u_3]))", anchors.spans, 6, rule));
    // A head that is no existence.
    ASSERT_FALSE(analyze(ea.definitions, "(>[1](in[1,u_1])(>[2](in[2,u_1])(in3[1,2,1,u_4])))",
        anchors.spans, 6, rule));
    // An operator input that is not a typed variable.
    ASSERT_FALSE(analyze(ea.definitions,
        "(>[1](in[1,u_1])(>[2](in[2,u_1])(existence1[u_1,1,u_5,u_4])))", anchors.spans, 6, rule));
}

TEST(demand_built_rules, the_witness_template_matches_concrete_demands_only) {
    gl::tests::ProverFixture ea("Peano");
    const PeanoAnchors anchors;
    gl::DemandBuiltRule rule;
    ASSERT_TRUE(analyze(ea.definitions, kSumRule, anchors.spans, 6, rule));
    gl::StrSpan values[gl::kDemandBuiltBoundCap];
    ASSERT_TRUE(gl::matchDemandBuiltWitness(rule, sp("(in3[a,b,marker,4])"), values));
    ASSERT_TRUE(gl::equalSpans(values[0], sp("a")) && gl::equalSpans(values[1], sp("b")));
    ASSERT_FALSE(gl::matchDemandBuiltWitness(rule, sp("(in3[a,b,marker,5])"), values));
    ASSERT_FALSE(gl::matchDemandBuiltWitness(rule, sp("(in3[a,marker,c,4])"), values));
    ASSERT_FALSE(gl::matchDemandBuiltWitness(rule, sp("(in3[u_a,b,marker,4])"), values));
    ASSERT_FALSE(gl::matchDemandBuiltWitness(rule, sp("(in2[a,marker,4])"), values));
    ASSERT_FALSE(gl::matchDemandBuiltWitness(rule, sp("!(in3[a,b,marker,4])"), values));
}

TEST(demand_built_rules, instantiation_spells_the_firing_texts) {
    gl::tests::ProverFixture ea("Peano");
    const PeanoAnchors anchors;
    gl::DemandBuiltRule rule;
    ASSERT_TRUE(analyze(ea.definitions, kSumRule, anchors.spans, 6, rule));
    const gl::StrSpan values[2] = { sp("a"), sp("b") };
    char out[gl::kDemandBuiltTextCap];
    const int32_t premiseLength = gl::instantiateDemandBuiltPart(rule, rule.premises[0], values, out,
                                                                 gl::kDemandBuiltTextCap);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(out, premiseLength), sp("(in[a,1])")));
    const int32_t headLength = gl::instantiateDemandBuiltPart(rule, rule.head, values, out,
                                                              gl::kDemandBuiltTextCap);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(out, headLength), sp("(existence1[1,a,b,4])")));
}

TEST(demand_built_rules, the_registry_holds_a_text_once_and_recognises_its_head) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    const PeanoAnchors anchors;
    gl::DemandBuiltRule rule;
    ASSERT_TRUE(analyze(ea.definitions, kSumRule, anchors.spans, 6, rule));
    gl::DemandBuiltRuleRegistry& registry = gl::DemandBuiltRuleRegistry::instance();
    registry.add(rule);
    registry.add(rule);
    ASSERT_EQ(registry.count(), 1);
    ASSERT_TRUE(gl::equalSpans(registry.at(0).textSpan(), gl::StrSpan(kSumRule)));
    // The head itself, and a premise whose own variables stand at the set or
    // the operator position: both match the head's instances.
    ASSERT_TRUE(registry.citesHead(sp("(existence1[u_1,9,10,u_4])")));
    ASSERT_TRUE(registry.citesHead(sp("(existence1[u_1,u_7,10,u_4])")));
    ASSERT_TRUE(registry.citesHead(sp("(existence1[u_1,x,y,f])")));
    ASSERT_TRUE(registry.citesHead(sp("(existence1[1,x,y,4])")));
    // Another fixed operator or set, a negation, another core.
    ASSERT_FALSE(registry.citesHead(sp("(existence1[u_1,x,y,u_5])")));
    ASSERT_FALSE(registry.citesHead(sp("(existence1[u_7,x,y,u_4])")));
    ASSERT_FALSE(registry.citesHead(sp("!(existence1[u_1,x,y,u_4])")));
    ASSERT_FALSE(registry.citesHead(sp("(in3[x,y,z,u_4])")));
}

TEST(demand_built_rules, the_install_decision_defers_and_registers_the_rule) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    mb.setExprKey("(" + ea.anchorInfo.name + "[1,2,3,4,5,6])");
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(kSumRule), gl::StrSpan(kSumCarrier)));
    ASSERT_EQ(gl::DemandBuiltRuleRegistry::instance().count(), 1);
    ASSERT_FALSE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb,
        sp("(>[1](in[1,u_1])(>[2](in2[2,u_1,u_3])(existence1[u_1,1,2,u_4])))"),
        gl::StrSpan(kSumCarrier)));
    ASSERT_EQ(gl::DemandBuiltRuleRegistry::instance().count(), 1);
    // A block without the anchor on its chain defers nothing.
    gl::Memory other;
    ASSERT_FALSE(ea.rejectedSourceReplay.deferDemandBuiltRule(other, gl::StrSpan(kSumRule), gl::StrSpan(kSumCarrier)));
}

TEST(demand_built_rules, a_demand_builds_the_instance_the_firing_would_deposit) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    prepareBlock(ea, mb, true);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(kSumRule), gl::StrSpan(kSumCarrier)));
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    ea.rejectedSourceReplay.buildDemandedInstances(mb, sp("(in3[a,b,marker,4])"), sp("main"));
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 1);
    ASSERT_TRUE(mb.nameMap.lookup("(existence1[1,a,b,4])") != 0);
    // A repeated premise instance, an untyped input, another operator: nothing.
    ea.rejectedSourceReplay.buildDemandedInstances(mb, sp("(in3[a,a,marker,4])"), sp("main"));
    ea.rejectedSourceReplay.buildDemandedInstances(mb, sp("(in3[a,c,marker,4])"), sp("main"));
    ea.rejectedSourceReplay.buildDemandedInstances(mb, sp("(in3[a,b,marker,5])"), sp("main"));
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 1);
    // A known instance builds nothing.
    const gl::NameId head = mb.nameMap.lookup("(existence1[1,a,b,4])");
    mb.intKnownStatements.insert(gl::StatementKey{ head, gl::NameMap::MAIN_ID }, gl::StatementFlags{ true, false });
    ea.rejectedSourceReplay.buildDemandedInstances(mb, sp("(in3[a,b,marker,4])"), sp("main"));
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 1);
}

TEST(demand_built_rules, a_block_without_the_rule_builds_nothing) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    prepareBlock(ea, mb, false);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(kSumRule), gl::StrSpan(kSumCarrier)));
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    ea.rejectedSourceReplay.buildDemandedInstances(mb, sp("(in3[a,b,marker,4])"), sp("main"));
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before);
}

TEST(demand_built_rules, a_waiting_demand_is_served_once_the_rule_arrives) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    prepareBlock(ea, mb, false);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(kSumRule), gl::StrSpan(kSumCarrier)));
    // The demand is filed while the block does not hold the rule yet.
    gl::SealedPageSet pages;
    pages.bind(&gl::staticMemory());
    const auto seal = [&pages](const std::string& v) {
        return gl::SealedString::copyFrom(pages, v.data(), static_cast<int32_t>(v.size()));
    };
    gl::StagedAdmissionValue value;
    const gl::SealedString keyElements[1] = { seal("(in3[a,b,c,4])") };
    value.key = gl::SealedSpan<gl::SealedString>::copyFrom(pages, keyElements, 1);
    mb.admissionKeysAlgebra.push_back(gl::AdmissionKeyAlgebraRecord{
        gl::SealedExpressionWithValidity{ seal("(in3[a,b,marker,4])"), seal("main") }, value });
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    ea.admissionMemory.drainAdmissionKeysAlgebra(mb);
    mb.admissionKeysAlgebra.clear();
    pages.seal();
    pages.freePages();
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before);
    // The rule arrives; the next burst's scan serves the waiting demand.
    mb.expandedImplications.mint(gl::LbStatePairKey{ mb.lbStateInterner.encode(kSumRule),
                                                     mb.lbStateInterner.encode("main") });
    ea.rejectedSourceReplay.buildWaitingInstances(mb);
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 1);
    ASSERT_TRUE(mb.nameMap.lookup("(existence1[1,a,b,4])") != 0);
}

TEST(demand_built_rules, analyze_accepts_a_typed_rule_with_an_or_head) {
    gl::tests::ProverFixture ea("Peano");
    const PeanoAnchors anchors;
    const std::string orCompact =
        ea.structuralCompiler.compileExpression("!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9]))");
    const std::string rule = typedOrRule(ea, orCompact);
    gl::DemandBuiltRule parsed;
    ASSERT_TRUE(gl::analyzeDemandBuiltRule(gl::StrSpan(rule), sp("(implication9[])"), ea.definitions,
        gl::StrSpan(ea.anchorInfo.name), anchors.spans, 6, parsed));
    ASSERT_TRUE(parsed.kind == gl::DemandBuiltKind::OrHead);
    ASSERT_EQ(parsed.premiseCount, 3);
    ASSERT_EQ(parsed.boundCount, 8);
    // The anchor slots carry their anchor values, the typed variables none.
    ASSERT_TRUE(gl::equalSpans(parsed.fixedSpan(0), sp("1")) && gl::equalSpans(parsed.fixedSpan(3), sp("4")));
    ASSERT_TRUE(gl::equalSpans(parsed.sliceSpan(parsed.bound[6]), sp("9")) && parsed.fixedSpan(6).len == 0);
    ASSERT_TRUE(gl::equalSpans(parsed.sliceSpan(parsed.head), gl::StrSpan(orCompact)));
    const auto refutations = templatesOf(parsed, gl::DemandBuiltTrigger::Refutation);
    const auto demands = templatesOf(parsed, gl::DemandBuiltTrigger::CaseDemand);
    const auto goals = templatesOf(parsed, gl::DemandBuiltTrigger::Goal);
    ASSERT_EQ(refutations.size(), std::size_t(2));
    ASSERT_TRUE(holds(refutations, "!(preorder[1,4,9,10])") && holds(refutations, "!(preorder[1,4,10,9])"));
    ASSERT_EQ(demands.size(), std::size_t(2));
    ASSERT_TRUE(holds(demands, "(preorder[1,4,9,10])") && holds(demands, "(preorder[1,4,10,9])"));
    ASSERT_TRUE(goals.size() == std::size_t(1) && goals[0] == orCompact);
    // An OR record has no witness template: the operator demands never match it.
    gl::StrSpan values[gl::kDemandBuiltBoundCap];
    ASSERT_FALSE(gl::matchDemandBuiltWitness(parsed, sp("(in3[a,b,marker,4])"), values));
}

TEST(demand_built_rules, an_and_disjunct_adds_the_refuters_of_its_conjuncts) {
    gl::tests::ProverFixture ea("Peano");
    const PeanoAnchors anchors;
    const std::string less = ea.structuralCompiler.compileExpression("(&(preorder[1,4,9,10])!(=[9,10]))");
    const std::string greater = ea.structuralCompiler.compileExpression("(&(preorder[1,4,10,9])!(=[10,9]))");
    ea.supportingRuleCompiler.compileConjunctionRefutations();
    const std::string orCompact = ea.structuralCompiler.compileExpression(
        "!(&(&!" + less + "!(=[9,10]))!" + greater + ")");
    gl::DemandBuiltRule parsed;
    ASSERT_TRUE(gl::analyzeDemandBuiltRule(gl::StrSpan(typedOrRule(ea, orCompact)), sp("(implication9[])"),
        ea.definitions, gl::StrSpan(ea.anchorInfo.name), anchors.spans, 6, parsed));
    const auto refutations = templatesOf(parsed, gl::DemandBuiltTrigger::Refutation);
    // The disjuncts' complements and, through the and disjuncts' refutation
    // compacts, their conjuncts' complements.
    ASSERT_TRUE(holds(refutations, "!" + less) && holds(refutations, "!" + greater));
    ASSERT_TRUE(holds(refutations, "!(preorder[1,4,9,10])") && holds(refutations, "!(preorder[1,4,10,9])"));
    ASSERT_TRUE(holds(refutations, "(=[9,10])") || holds(refutations, "(=[10,9])"));
    ASSERT_TRUE(holds(refutations, "!(=[9,10])") || holds(refutations, "!(=[10,9])"));
    ASSERT_TRUE(holds(templatesOf(parsed, gl::DemandBuiltTrigger::CaseDemand), less));
    ASSERT_EQ(templatesOf(parsed, gl::DemandBuiltTrigger::Goal).size(), std::size_t(1));
}

TEST(demand_built_rules, analyze_rejects_or_heads_outside_the_criterion) {
    gl::tests::ProverFixture ea("Peano");
    const PeanoAnchors anchors;
    gl::DemandBuiltRule parsed;
    const auto accepts = [&](const std::string& rule) {
        return gl::analyzeDemandBuiltRule(gl::StrSpan(rule), sp("(implication9[])"), ea.definitions,
            gl::StrSpan(ea.anchorInfo.name), anchors.spans, 6, parsed);
    };
    const std::string anchor = "(" + ea.anchorInfo.name + "[1,2,3,4,5,6])";
    const std::string pairOr =
        ea.structuralCompiler.compileExpression("!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9]))");
    // One typed variable: a single-variable typed OR rule stays eager.
    const std::string singleOr = ea.structuralCompiler.compileExpression("!(&!(=[9,2])!(preorder[1,4,2,9]))");
    ASSERT_FALSE(accepts("(>[1,2,3,4,5,6]" + anchor + "(>[9](in[9,1])" + singleOr + "))"));
    // A disjunct that does not name every typed variable.
    const std::string partialOr = ea.structuralCompiler.compileExpression("!(&!(=[9,2])!(preorder[1,4,9,10]))");
    ASSERT_FALSE(accepts(typedOrRule(ea, partialOr)));
    // A premise other than a typing.
    ASSERT_FALSE(accepts("(>[1,2,3,4,5,6]" + anchor + "(>[9](in[9,1])(>[10](in2[10,1,3])" + pairOr + ")))"));
    // A typing set that is neither an anchor slot nor an anchor argument.
    ASSERT_FALSE(accepts("(>[1,2,3,4,5,6]" + anchor + "(>[7](in[7,1])(>[9](in[9,7])(>[10](in[10,1])"
        + pairOr + "))))"));
    // A block without anchor arguments.
    ASSERT_FALSE(gl::analyzeDemandBuiltRule(gl::StrSpan(typedOrRule(ea, pairOr)), sp("(implication9[])"),
        ea.definitions, gl::StrSpan(ea.anchorInfo.name), anchors.spans, 0, parsed));
    ASSERT_TRUE(accepts(typedOrRule(ea, pairOr)));
}

TEST(demand_built_rules, an_or_trigger_matches_concrete_statements_only) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    const PeanoAnchors anchors;
    const std::string orCompact =
        ea.structuralCompiler.compileExpression("!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9]))");
    gl::DemandBuiltRule parsed;
    ASSERT_TRUE(gl::analyzeDemandBuiltRule(gl::StrSpan(typedOrRule(ea, orCompact)), sp("(implication9[])"),
        ea.definitions, gl::StrSpan(ea.anchorInfo.name), anchors.spans, 6, parsed));
    int32_t refutation = -1;
    for (int32_t t = 0; t < parsed.triggerCount; ++t)
        if (gl::equalSpans(parsed.triggerSpan(t), sp("!(preorder[1,4,9,10])"))) refutation = t;
    ASSERT_TRUE(refutation >= 0);
    gl::StrSpan values[gl::kDemandBuiltBoundCap];
    ASSERT_TRUE(gl::matchDemandBuiltTemplate(parsed, refutation, sp("!(preorder[1,4,a,b])"), values));
    ASSERT_TRUE(gl::equalSpans(values[6], sp("a")) && gl::equalSpans(values[7], sp("b")));
    ASSERT_TRUE(gl::equalSpans(values[0], sp("1")) && gl::equalSpans(values[3], sp("4")));
    // Another anchor operator, the other polarity, a template argument,
    // another arity, another core.
    ASSERT_FALSE(gl::matchDemandBuiltTemplate(parsed, refutation, sp("!(preorder[1,5,a,b])"), values));
    ASSERT_FALSE(gl::matchDemandBuiltTemplate(parsed, refutation, sp("(preorder[1,4,a,b])"), values));
    ASSERT_FALSE(gl::matchDemandBuiltTemplate(parsed, refutation, sp("!(preorder[1,4,u_a,b])"), values));
    ASSERT_FALSE(gl::matchDemandBuiltTemplate(parsed, refutation, sp("!(preorder[1,4,a])"), values));
    ASSERT_FALSE(gl::matchDemandBuiltTemplate(parsed, refutation, sp("!(in2[1,a,b])"), values));
    // The matched values instantiate the firing's premises and head.
    ASSERT_TRUE(gl::matchDemandBuiltTemplate(parsed, refutation, sp("!(preorder[1,4,a,b])"), values));
    char out[gl::kDemandBuiltTextCap];
    const int32_t anchorLength = gl::instantiateDemandBuiltPart(parsed, parsed.premises[0], values, out,
                                                                gl::kDemandBuiltTextCap);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(out, anchorLength),
        gl::StrSpan("(" + ea.anchorInfo.name + "[1,2,3,4,5,6])")));
    const int32_t typingLength = gl::instantiateDemandBuiltPart(parsed, parsed.premises[2], values, out,
                                                                gl::kDemandBuiltTextCap);
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(out, typingLength), sp("(in[b,1])")));
    const int32_t headLength = gl::instantiateDemandBuiltPart(parsed, parsed.head, values, out,
                                                              gl::kDemandBuiltTextCap);
    const std::string head(out, static_cast<std::size_t>(headLength));
    ASSERT_TRUE(head.find("[1,4,a,b]") != std::string::npos || head.find("[1,4,b,a]") != std::string::npos);
    // The registry recognises a premise that could match the OR head.
    gl::DemandBuiltRuleRegistry& registry = gl::DemandBuiltRuleRegistry::instance();
    registry.add(parsed);
    const std::string core = orCompact.substr(0, orCompact.find('['));
    ASSERT_TRUE(registry.citesHead(gl::StrSpan(core + "[u_1,u_4,x,y])")));
    ASSERT_TRUE(registry.citesHead(gl::StrSpan(core + "[n,f,x,y])")));
    ASSERT_FALSE(registry.citesHead(gl::StrSpan(core + "[u_1,u_5,x,y])")));
}

TEST(demand_built_rules, rename_maps_every_argument_and_writes_the_complement_on_request) {
    const gl::StrSpan from[3] = { sp("u_1"), sp("u_3"), sp("u_4") };
    const gl::StrSpan to[3] = { sp("1"), sp("9"), sp("10") };
    char out[gl::kDemandBuiltTextCap];
    int32_t length = 0;
    ASSERT_TRUE(gl::demandBuiltRename(sp("(preorder[u_1,u_1,u_3,u_4])"), from, to, 3, false, out,
                                      gl::kDemandBuiltTextCap, length));
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(out, length), sp("(preorder[1,1,9,10])")));
    ASSERT_TRUE(gl::demandBuiltRename(sp("(=[u_3,u_4])"), from, to, 3, true, out, gl::kDemandBuiltTextCap, length));
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(out, length), sp("!(=[9,10])")));
    ASSERT_TRUE(gl::demandBuiltRename(sp("!(=[u_3,u_4])"), from, to, 3, true, out, gl::kDemandBuiltTextCap, length));
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(out, length), sp("(=[9,10])")));
    ASSERT_TRUE(gl::demandBuiltRename(sp("!(=[u_3,u_4])"), from, to, 3, false, out, gl::kDemandBuiltTextCap, length));
    ASSERT_TRUE(gl::equalSpans(gl::StrSpan(out, length), sp("!(=[9,10])")));
    // A token without a target, and a nested argument: the renaming does not cover them.
    ASSERT_FALSE(gl::demandBuiltRename(sp("(preorder[u_1,u_2,u_3,u_4])"), from, to, 3, false, out,
                                       gl::kDemandBuiltTextCap, length));
    ASSERT_FALSE(gl::demandBuiltRename(sp("(f[u_1,(g[u_3])])"), from, to, 3, false, out,
                                       gl::kDemandBuiltTextCap, length));
}

namespace {

/// A block on the Peano anchor holding the typed OR rule over a two-preorder
/// OR at main: the anchor fact, the carrier and the typings of @p typed are
/// known there. Returns the rule text.
std::string prepareOrBlock(gl::tests::ProverFixture& ea, gl::Memory& mb,
                           const std::vector<std::string>& typed, bool holdRule) {
    const std::string anchor = "(" + ea.anchorInfo.name + "[1,2,3,4,5,6])";
    mb.setExprKey(anchor);
    const std::string orCompact =
        ea.structuralCompiler.compileExpression("!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9]))");
    const std::string rule = typedOrRule(ea, orCompact);
    const int level = 0;
    std::vector<std::string> known = { anchor, "(implication9[])" };
    for (const std::string& value : typed) known.push_back("(in[" + value + ",1])");
    for (const std::string& text : known) {
        const gl::NameId id = mb.nameMap.encode(text);
        mb.intKnownStatements.insert(gl::StatementKey{ id, gl::NameMap::MAIN_ID }, gl::StatementFlags{ true, false });
        mb.intStatementLevelsMap.assignSet(gl::StatementKey{ id, gl::NameMap::MAIN_ID }, &level, 1);
    }
    if (holdRule)
        mb.expandedImplications.mint(gl::LbStatePairKey{ mb.lbStateInterner.encode(rule),
                                                         mb.lbStateInterner.encode("main") });
    return rule;
}

/// Make @p text known at main and return its packed key.
int64_t knowAtMain(gl::Memory& mb, const std::string& text) {
    const int level = 0;
    const gl::NameId id = mb.nameMap.encode(text);
    mb.intKnownStatements.insert(gl::StatementKey{ id, gl::NameMap::MAIN_ID }, gl::StatementFlags{ true, false });
    mb.intStatementLevelsMap.assignSet(gl::StatementKey{ id, gl::NameMap::MAIN_ID }, &level, 1);
    return gl::packStatementKey(id, gl::NameMap::MAIN_ID);
}

/// The instance text of the OR head over @p left and @p right.
std::string orInstance(const std::string& rule, const std::string& left, const std::string& right) {
    const std::size_t start = rule.rfind("(or");
    const std::string core = rule.substr(start, rule.find('[', start) - start);
    return core + "[1,4," + left + "," + right + "])";
}

}

TEST(demand_built_rules, shape_and_registry_probes_recognise_or_triggers) {
    ASSERT_TRUE(gl::demandBuiltSameShape(sp("!(preorder[1,4,9,10])"), sp("!(preorder[1,4,a,b])")));
    ASSERT_FALSE(gl::demandBuiltSameShape(sp("!(preorder[1,4,9,10])"), sp("(preorder[1,4,a,b])")));
    ASSERT_FALSE(gl::demandBuiltSameShape(sp("!(preorder[1,4,9,10])"), sp("!(preorder[1,a,b])")));
    ASSERT_FALSE(gl::demandBuiltSameShape(sp("(=[9,10])"), sp("(in[9,10])")));
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::DemandBuiltRuleRegistry& registry = gl::DemandBuiltRuleRegistry::instance();
    ASSERT_FALSE(registry.hasOrRules());
    gl::Memory& mb = ea.body;
    const std::string rule = prepareOrBlock(ea, mb, {}, false);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(rule), sp("(implication9[])")));
    ASSERT_TRUE(registry.hasOrRules());
    const gl::DemandBuiltRule* found = registry.find(gl::StrSpan(rule));
    ASSERT_TRUE(found != nullptr && found->kind == gl::DemandBuiltKind::OrHead);
    ASSERT_TRUE(registry.find(sp("(>[1](in[1,u_1])(in[1,u_1]))")) == nullptr);
    ASSERT_TRUE(gl::demandBuiltHasTrigger(*found, sp("!(preorder[1,4,x,y])"), gl::DemandBuiltTrigger::Refutation));
    ASSERT_FALSE(gl::demandBuiltHasTrigger(*found, sp("(preorder[1,4,x,y])"), gl::DemandBuiltTrigger::Refutation));
    ASSERT_TRUE(gl::demandBuiltHasTrigger(*found, sp("(preorder[1,4,x,y])"), gl::DemandBuiltTrigger::CaseDemand));
    ASSERT_TRUE(registry.hasOrTrigger(sp("!(preorder[1,4,x,y])"), gl::DemandBuiltTrigger::Refutation));
    ASSERT_FALSE(registry.hasOrTrigger(sp("!(in[x,1])"), gl::DemandBuiltTrigger::Refutation));
}

TEST(demand_built_rules, a_refuting_fact_builds_both_or_instances_of_its_pair) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    const std::string rule = prepareOrBlock(ea, mb, { "a", "b" }, true);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(rule), sp("(implication9[])")));
    // An unrelated row is no event; the refuter is.
    gl::StatementArrivals::stageDeferredOrEvent(mb, knowAtMain(mb, "(in2[a,b,3])"));
    ASSERT_EQ(mb.pendingDeferredOrEvents.size(), 0);
    gl::StatementArrivals::stageDeferredOrEvent(mb, knowAtMain(mb, "!(preorder[1,4,a,b])"));
    ASSERT_EQ(mb.pendingDeferredOrEvents.size(), 1);
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    ea.rejectedSourceReplay.drainDeferredOrEvents(mb, mb.sameIterationInternalMail);
    ASSERT_EQ(mb.pendingDeferredOrEvents.size(), 0);
    // The refuter refutes a disjunct of the instance over (a, b) and of the one
    // over (b, a): the eager rule fires both.
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 2);
    ASSERT_TRUE(mb.nameMap.lookup(orInstance(rule, "a", "b")) != 0);
    ASSERT_TRUE(mb.nameMap.lookup(orInstance(rule, "b", "a")) != 0);
    ASSERT_TRUE(mb.deferredOrWaiting.empty());
}

TEST(demand_built_rules, a_refuter_with_an_untyped_value_waits_for_its_typing) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    const std::string rule = prepareOrBlock(ea, mb, { "a" }, true);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(rule), sp("(implication9[])")));
    gl::StatementArrivals::stageDeferredOrEvent(mb, knowAtMain(mb, "!(preorder[1,4,a,c])"));
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    ea.rejectedSourceReplay.drainDeferredOrEvents(mb, mb.sameIterationInternalMail);
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before);
    const gl::NameId typing = mb.nameMap.lookup("(in[c,1])");
    ASSERT_TRUE(typing != 0 && mb.deferredOrWaiting.lookup(static_cast<int32_t>(typing)) != 0);
    // The typing arrives: it is staged as the awaited premise and re-offers
    // the refuter, which now builds both instances.
    gl::StatementArrivals::stageDeferredOrEvent(mb, knowAtMain(mb, "(in[c,1])"));
    ASSERT_EQ(mb.pendingDeferredOrEvents.size(), 1);
    ea.rejectedSourceReplay.drainDeferredOrEvents(mb, mb.sameIterationInternalMail);
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 2);
    ASSERT_TRUE(mb.nameMap.lookup(orInstance(rule, "a", "c")) != 0);
    ASSERT_TRUE(mb.deferredOrWaiting.empty());
}

TEST(demand_built_rules, a_newly_held_or_rule_stages_the_known_refuters) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    const std::string rule = prepareOrBlock(ea, mb, { "a", "b" }, false);
    const int64_t refuter = knowAtMain(mb, "!(preorder[1,4,a,b])");
    knowAtMain(mb, "(in2[a,b,3])");
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(rule), sp("(implication9[])")));
    // Without the rule held, the drain builds nothing and nothing waits.
    gl::StatementArrivals::stageDeferredOrEvent(mb, refuter);
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    ea.rejectedSourceReplay.drainDeferredOrEvents(mb, mb.sameIterationInternalMail);
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before);
    ASSERT_TRUE(mb.deferredOrWaiting.empty());
    // The rule arrives: the scan stages the known refuter only.
    mb.expandedImplications.mint(gl::LbStatePairKey{ mb.lbStateInterner.encode(rule),
                                                     mb.lbStateInterner.encode("main") });
    ea.rejectedSourceReplay.stageExistingDeferredOrEvents(mb, gl::StrSpan(rule));
    ASSERT_EQ(mb.pendingDeferredOrEvents.size(), 1);
    ASSERT_EQ(mb.pendingDeferredOrEvents[0], refuter);
    ea.rejectedSourceReplay.drainDeferredOrEvents(mb, mb.sameIterationInternalMail);
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 2);
}

TEST(demand_built_rules, or_rules_are_visited_in_rule_text_order) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    const PeanoAnchors anchors;
    gl::DemandBuiltRuleRegistry& registry = gl::DemandBuiltRuleRegistry::instance();
    gl::DemandBuiltRule sum;
    ASSERT_TRUE(analyze(ea.definitions, kSumRule, anchors.spans, 6, sum));
    const std::string pairOr =
        ea.structuralCompiler.compileExpression("!(&!(preorder[1,4,9,10])!(preorder[1,4,10,9]))");
    const std::string swappedOr =
        ea.structuralCompiler.compileExpression("!(&!(preorder[1,4,10,9])!(=[9,10]))");
    gl::DemandBuiltRule first;
    gl::DemandBuiltRule second;
    ASSERT_TRUE(gl::analyzeDemandBuiltRule(gl::StrSpan(typedOrRule(ea, pairOr)), sp("(implication9[])"),
        ea.definitions, gl::StrSpan(ea.anchorInfo.name), anchors.spans, 6, first));
    ASSERT_TRUE(gl::analyzeDemandBuiltRule(gl::StrSpan(typedOrRule(ea, swappedOr)), sp("(implication9[])"),
        ea.definitions, gl::StrSpan(ea.anchorInfo.name), anchors.spans, 6, second));
    const bool firstSmaller = gl::compareSpans(first.textSpan(), second.textSpan()) < 0;
    registry.add(firstSmaller ? second : first);
    registry.add(sum);
    registry.add(firstSmaller ? first : second);
    int32_t order[gl::kDemandBuiltRuleCap];
    ASSERT_EQ(registry.orderedOrRules(order), 2);
    ASSERT_TRUE(gl::compareSpans(registry.at(order[0]).textSpan(), registry.at(order[1]).textSpan()) < 0);
    ASSERT_TRUE(registry.at(order[0]).kind == gl::DemandBuiltKind::OrHead);
}

TEST(demand_built_rules, a_goal_equal_to_an_instance_builds_that_instance) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    const std::string rule = prepareOrBlock(ea, mb, { "a", "b" }, false);
    // The goal is filed before the block holds the rule.
    const gl::NameId goalId = mb.nameMap.encode(orInstance(rule, "a", "b"));
    const int64_t goal = gl::packStatementKey(goalId, gl::NameMap::MAIN_ID);
    mb.intToBeProved.assignSet(goal, nullptr, 0);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(rule), sp("(implication9[])")));
    mb.expandedImplications.mint(gl::LbStatePairKey{ mb.lbStateInterner.encode(rule),
                                                     mb.lbStateInterner.encode("main") });
    ea.rejectedSourceReplay.stageExistingDeferredOrEvents(mb, gl::StrSpan(rule));
    ASSERT_EQ(mb.pendingDeferredOrEvents.size(), 1);
    ASSERT_EQ(mb.pendingDeferredOrEvents[0], goal);
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    ea.rejectedSourceReplay.drainDeferredOrEvents(mb, mb.sameIterationInternalMail);
    // Only the goal's own instance.
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 1);
    ASSERT_TRUE(mb.nameMap.lookup(orInstance(rule, "b", "a")) == 0);
}

TEST(demand_built_rules, a_case_demand_builds_the_instances_of_its_disjunct) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    const std::string rule = prepareOrBlock(ea, mb, { "a", "b" }, true);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(rule), sp("(implication9[])")));
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    // No OR head has such a disjunct: nothing.
    ea.rejectedSourceReplay.buildDemandedOrCaseInstances(mb, sp("(in2[a,b,3])"), sp("main"));
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before);
    // The disjunct a <= b of the instance over (a, b) and of the one over (b, a).
    ea.rejectedSourceReplay.buildDemandedOrCaseInstances(mb, sp("(preorder[1,4,a,b])"), sp("main"));
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 2);
    ASSERT_TRUE(mb.nameMap.lookup(orInstance(rule, "a", "b")) != 0);
    ASSERT_TRUE(mb.nameMap.lookup(orInstance(rule, "b", "a")) != 0);
}

TEST(demand_built_rules, a_stored_case_demand_is_served_once_the_rule_arrives) {
    gl::tests::ProverFixture ea("Peano");
    const RegistryGuard guard;
    gl::Memory& mb = ea.body;
    const std::string rule = prepareOrBlock(ea, mb, { "a", "b" }, false);
    ASSERT_TRUE(ea.rejectedSourceReplay.deferDemandBuiltRule(mb, gl::StrSpan(rule), sp("(implication9[])")));
    gl::ScratchArena& arena = gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    const int64_t demand = gl::mintTemplateKey(mb.templateInterner, mb.nameMap,
        std::string("(preorder[1,4,a,b])"), std::string("main"));
    gl::insertAdmissionOrdis2IdsBlob(mb.overallHashMemory.admissionMapOrdis2, demand, 0, nullptr, 0,
        mb.ruleInterner, arena);
    const int32_t before = mb.sameIterationInternalMail.statements_.count();
    // Without the rule held the demand builds nothing; once held, the
    // next burst's re-offer builds both instances.
    ea.rejectedSourceReplay.buildWaitingInstances(mb);
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before);
    mb.expandedImplications.mint(gl::LbStatePairKey{ mb.lbStateInterner.encode(rule),
                                                     mb.lbStateInterner.encode("main") });
    ea.rejectedSourceReplay.buildWaitingInstances(mb);
    ASSERT_EQ(mb.sameIterationInternalMail.statements_.count(), before + 2);
}
