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
/// @brief Tests for the definition-set typing of `=` and the LB records
///        (I-267).
///
/// @details
/// Covers the union-find table, the leaf typing rule (a slot fixes, `=`
/// joins, an unfixed group takes `=`'s configured label), typing through a
/// compiled compact without a config entry, the LB key parser, the per-LB
/// variable definition-set records written at registration (chain,
/// contradiction and recursion LBs, shared-prefix agreement) and on
/// compressor nodes, and `findDigitArgs` counting an `=` argument only when
/// it is typed `(1)`. The typed-equality integration: the extensionality
/// entities (compact reuse, relation shapes, subset orientation), their
/// registration from a theorem's head, the on-the-spot goal typing (LB
/// records, subproof element through its interface slots), and a set goal
/// opening its two subset subproofs while a digit goal stays atomic.

#include "test_harness.hpp"

#include "../prover.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace {

    /// The supplied lemma "two intervals with the same bounds are equal"
    /// (slots 9 n, 10 a, 11 b).
    const char* const kRow84 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])"
        "(>[11](interval[1,4,2,9,11])(=[10,11]))))";

    /// Label of one variable after typing @p leaves with @p ea.
    std::string labelAfter(const gl::ExpressionAnalyzer& ea,
                           const std::vector<std::string>& leaves,
                           const std::string& var) {
        gl::ExpressionAnalyzer::VarTypeTable table;
        for (const std::string& leaf : leaves) ea.typeLeafInto(gl::StrSpan(leaf), table);
        const int32_t node = table.indexOf(gl::StrSpan(var));
        if (node < 0) return std::string("<absent>");
        return ea.resolvedLabel(table, node).toStdString();
    }

    /// The LB reached from the root through the given key chain.
    gl::Memory* childAt(gl::ExpressionAnalyzer& ea,
                        const std::vector<std::string>& keys) {
        gl::Memory* m = &ea.body;
        for (const std::string& key : keys) {
            m = ea.simpleMapStore.findChild(m, key);
            if (m == nullptr) return nullptr;
        }
        return m;
    }

} // namespace

TEST(typed_equality, var_type_table_fix_join_and_labels) {
    gl::ExpressionAnalyzer::VarTypeTable t;
    const int32_t a = t.add(gl::StrSpan("a", 1));
    const int32_t b = t.add(gl::StrSpan("b", 1));
    const int32_t c = t.add(gl::StrSpan("c", 1));
    ASSERT_EQ(t.add(gl::StrSpan("a", 1)), a);          // idempotent
    ASSERT_EQ(t.indexOf(gl::StrSpan("zz", 2)), -1);    // absent name
    ASSERT_EQ(t.count, 3);

    t.fix(b, gl::StrSpan("P(1)", 4));
    ASSERT_TRUE(t.labelOf(a).empty());                 // still unfixed
    t.join(a, b);                                      // a joins b's fixed group
    ASSERT_EQ(t.labelOf(a).toStdString(), std::string("P(1)"));
    ASSERT_EQ(t.root(b), a);                           // smaller index is the root
    ASSERT_TRUE(t.labelOf(c).empty());
    t.join(c, b);
    ASSERT_EQ(t.labelOf(c).toStdString(), std::string("P(1)"));
    t.fix(c, gl::StrSpan("P(1)", 4));                  // same label: no conflict
    ASSERT_EQ(t.labelOf(a).toStdString(), std::string("P(1)"));
}

TEST(typed_equality, flat_operator_leaf_shapes) {
    using EA = gl::ExpressionAnalyzer;
    ASSERT_TRUE(EA::isFlatOperatorLeaf(gl::StrSpan(std::string("(in[1,2])"))));
    ASSERT_TRUE(EA::isFlatOperatorLeaf(gl::StrSpan(std::string("(=[a,b])"))));
    ASSERT_FALSE(EA::isFlatOperatorLeaf(gl::StrSpan(std::string("in[1,2]"))));
    ASSERT_FALSE(EA::isFlatOperatorLeaf(gl::StrSpan(std::string("(>[x](in[x,A])(in[x,B]))"))));
    ASSERT_FALSE(EA::isFlatOperatorLeaf(gl::StrSpan(std::string("(=[s(x),y])"))));
    ASSERT_FALSE(EA::isFlatOperatorLeaf(gl::StrSpan(std::string("!(in[1,2])"))));
}

TEST(typed_equality, set_equality_types_through_interval_slots) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    const std::vector<std::string> leaves = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(interval[1,4,2,9,10])",
        "(interval[1,4,2,9,11])",
        "(=[10,11])" };
    ASSERT_EQ(labelAfter(ea, leaves, "10"), std::string("P(1)"));
    ASSERT_EQ(labelAfter(ea, leaves, "11"), std::string("P(1)"));
    ASSERT_EQ(labelAfter(ea, leaves, "9"), std::string("(1)"));
    ASSERT_EQ(labelAfter(ea, leaves, "4"), std::string("P(x(1)(x(1)(1)))"));
    ASSERT_EQ(labelAfter(ea, leaves, "99"), std::string("<absent>"));
    // Leaf order does not matter: the equality typed first still resolves.
    const std::vector<std::string> reversed(leaves.rbegin(), leaves.rend());
    ASSERT_EQ(labelAfter(ea, reversed, "11"), std::string("P(1)"));
}

TEST(typed_equality, unfixed_equality_takes_the_start_label) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ASSERT_EQ(ea.equalityStartLabel().toStdString(), std::string("(1)"));
    const std::vector<std::string> leaves = { "(=[a,b])" };
    ASSERT_EQ(labelAfter(ea, leaves, "a"), std::string("(1)"));
    ASSERT_EQ(labelAfter(ea, leaves, "b"), std::string("(1)"));
    // A digit equality stays a digit equality.
    const std::vector<std::string> digits = { "(in2[7,8,3])", "(=[7,8])" };
    ASSERT_EQ(labelAfter(ea, digits, "8"), std::string("(1)"));
}

TEST(typed_equality, relation_equalities_type_through_membership_slots) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    const std::vector<std::string> binary = { "(in2[x,y,f])", "(=[f,g])" };
    ASSERT_EQ(labelAfter(ea, binary, "g"), std::string("P(x(1)(1))"));
    ASSERT_EQ(labelAfter(ea, binary, "x"), std::string("(1)"));
    // Polarity does not change types: a negated leaf types the same.
    const std::vector<std::string> ternary = { "!(in3[x,y,z,h])", "(=[k,h])" };
    ASSERT_EQ(labelAfter(ea, ternary, "k"), std::string("P(x(1)(x(1)(1)))"));
}

TEST(typed_equality, compact_without_config_types_through_its_elements) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    // A compact whose formals sit in set slots of its elements.
    ea.compiledExpressions.emplace("typedEqTestSubset", gl::LogicalEntity(
        "implication", { "(in[1,u_1])", "(in[1,u_2])" },
        "(typedEqTestSubset[u_1,u_2])", 2));
    // A compact whose formals are only joined by an equality.
    ea.compiledExpressions.emplace("typedEqTestJoin", gl::LogicalEntity(
        "and", { "(=[u_1,u_2])" }, "(typedEqTestJoin[u_1,u_2])", 2));
    ASSERT_TRUE(ea.coreConfig(gl::StrSpan(std::string("typedEqTestSubset"))) == nullptr);

    const std::vector<std::string> subset = { "(typedEqTestSubset[A,B])" };
    ASSERT_EQ(labelAfter(ea, subset, "A"), std::string("P(1)"));
    ASSERT_EQ(labelAfter(ea, subset, "B"), std::string("P(1)"));

    // The compact's internal join carries a label from outside it.
    const std::vector<std::string> joined = {
        "(typedEqTestJoin[S,T])", "(interval[1,4,2,9,S])" };
    ASSERT_EQ(labelAfter(ea, joined, "T"), std::string("P(1)"));
    // Without the outside label the joined pair takes the start label.
    const std::vector<std::string> alone = { "(typedEqTestJoin[S,T])" };
    ASSERT_EQ(labelAfter(ea, alone, "T"), std::string("(1)"));
}

TEST(typed_equality, leading_expression_of_lb_keys) {
    using EA = gl::ExpressionAnalyzer;
    const auto lead = [](const std::string& key) {
        return EA::leadingExpressionOfKey(gl::StrSpan(key)).toStdString();
    };
    ASSERT_EQ(lead("(interval[1,4,2,9,10])"), std::string("(interval[1,4,2,9,10])"));
    ASSERT_EQ(lead("__contradiction__(fold[1,3])"), std::string("(fold[1,3])"));
    ASSERT_EQ(lead("__contradiction__!(=[7,8])"), std::string("!(=[7,8])"));
    ASSERT_EQ(lead("(=[9,2])_induction_rec0_"), std::string("(=[9,2])"));
    ASSERT_EQ(lead("CompressorNode_3"), std::string(""));
    ASSERT_EQ(lead("12"), std::string(""));
    ASSERT_EQ(lead(""), std::string(""));
}

TEST(typed_equality, memory_record_set_view_and_lookup) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    gl::Memory* lb = ea.lbStore.create<gl::Memory>();
    ASSERT_TRUE(lb->varDefSetsView().empty());
    ASSERT_EQ(lb->varDefSetsId, 0);
    ASSERT_TRUE(gl::ExpressionAnalyzer::lookupLbVarDefinitionSet(
        *lb, gl::StrSpan("a", 1)).empty());               // no record at all

    lb->setVarDefSets(gl::StrSpan(std::string("a:(1);S:P(1);R:P(x(1)(1))")));
    ASSERT_NE(lb->varDefSetsId, 0);
    ASSERT_EQ(lb->varDefSetsView().toStdString(), std::string("a:(1);S:P(1);R:P(x(1)(1))"));
    const auto lookup = [&](const char* v) {
        return gl::ExpressionAnalyzer::lookupLbVarDefinitionSet(
            *lb, gl::StrSpan(std::string(v))).toStdString();
    };
    ASSERT_EQ(lookup("a"), std::string("(1)"));
    ASSERT_EQ(lookup("S"), std::string("P(1)"));
    ASSERT_EQ(lookup("R"), std::string("P(x(1)(1))"));
    ASSERT_EQ(lookup("Sx"), std::string(""));           // prefix of no name
    ASSERT_EQ(lookup("b"), std::string(""));            // not recorded
    ea.lbStore.destroy(lb);
}

TEST(typed_equality, registration_records_chain_and_contradiction_lbs) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.parameters.try_contradiction = true;
    ea.parameters.try_contradiction_negated_head = true;
    ea.addTheoremToMemory(std::string(kRow84), ea.body, 0, false, ea.globalDependencies);

    const std::string anchor = "(AnchorFTA[1,2,3,4,5,6,7,8])";
    gl::Memory* anchorLB = childAt(ea, { anchor });
    gl::Memory* firstLB = childAt(ea, { anchor, "(interval[1,4,2,9,10])" });
    gl::Memory* secondLB = childAt(ea, { anchor, "(interval[1,4,2,9,10])",
                                         "(interval[1,4,2,9,11])" });
    ASSERT_TRUE(anchorLB != nullptr && firstLB != nullptr && secondLB != nullptr);

    ASSERT_EQ(anchorLB->varDefSetsView().toStdString(), std::string(
        "1:P(1);2:(1);3:P(x(1)(1));4:P(x(1)(x(1)(1)));5:P(x(1)(x(1)(1)));"
        "6:(1);7:(1);8:P(x(1)(1))"));
    ASSERT_EQ(firstLB->varDefSetsView().toStdString(), std::string(
        "1:P(1);4:P(x(1)(x(1)(1)));2:(1);9:(1);10:P(1)"));
    ASSERT_EQ(gl::ExpressionAnalyzer::lookupLbVarDefinitionSet(
        *secondLB, gl::StrSpan("11", 2)).toStdString(), std::string("P(1)"));
    ASSERT_TRUE(ea.lbVarDefinitionSetsMatch(*secondLB));

    // Both contradiction LBs record the head's arguments as sets.
    gl::Memory* verbatim = ea.simpleMapStore.findChild(secondLB, "__contradiction__(=[10,11])");
    gl::Memory* negated = ea.simpleMapStore.findChild(secondLB, "__contradiction__!(=[10,11])");
    ASSERT_TRUE(verbatim != nullptr && negated != nullptr);
    ASSERT_EQ(verbatim->varDefSetsView().toStdString(), std::string("10:P(1);11:P(1)"));
    ASSERT_EQ(negated->varDefSetsView().toStdString(), std::string("10:P(1);11:P(1)"));

    // No induction over the sets: findDigitArgs drops the set-typed `=`
    // arguments, so no recursion block hangs under the innermost LB.
    ASSERT_TRUE(ea.simpleMapStore.findChild(secondLB, "(in2[rec0,10,3])") == nullptr);
    ASSERT_TRUE(ea.simpleMapStore.findChild(secondLB, "(in2[rec0,11,3])") == nullptr);
}

TEST(typed_equality, registration_records_recursion_lbs_and_shared_prefix) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.addTheoremToMemory(std::string(kRow84), ea.body, 0, false, ea.globalDependencies);
    // A second theorem sharing the prefix up to (interval[1,4,2,9,10]); its
    // digit 9 schedules induction (recursion blocks #1 and #2). Reaching the
    // shared LB re-checks its record (asserted inside the registration).
    const std::string shared =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](interval[1,4,2,9,10])"
        "(in[9,1])))";
    ea.addTheoremToMemory(shared, ea.body, 0, false, ea.globalDependencies);

    gl::Memory* firstLB = childAt(ea, { "(AnchorFTA[1,2,3,4,5,6,7,8])",
                                        "(interval[1,4,2,9,10])" });
    ASSERT_TRUE(firstLB != nullptr);
    ASSERT_TRUE(ea.lbVarDefinitionSetsMatch(*firstLB));

    gl::Memory* block1 = ea.simpleMapStore.findChild(firstLB, "(in2[rec0,9,3])");
    ASSERT_TRUE(block1 != nullptr);
    ASSERT_EQ(block1->varDefSetsView().toStdString(),
              std::string("rec0:(1);9:(1);3:P(x(1)(1))"));
    gl::Memory* block2 = ea.simpleMapStore.findChild(firstLB, "(=[s(rec0),2])");
    ASSERT_TRUE(block2 != nullptr);
    ASSERT_EQ(block2->varDefSetsView().toStdString(), std::string("9:(1);2:(1)"));
}

TEST(typed_equality, compressor_node_records_the_whole_theorem) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    gl::Memory* lb = ea.lbStore.create<gl::Memory>();
    lb->setExprKey("CompressorNode_0");
    const std::vector<std::string> leaves = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(interval[1,4,2,9,10])",
        "(interval[1,4,2,9,11])",
        "(=[10,11])" };
    gl::StrSpan spans[4];
    for (int i = 0; i < 4; ++i) spans[i] = gl::StrSpan(leaves[i]);
    ea.assignTheoremVarDefinitionSets(*lb, spans, 4);
    ASSERT_EQ(lb->varDefSetsView().toStdString(), std::string(
        "1:P(1);2:(1);3:P(x(1)(1));4:P(x(1)(x(1)(1)));5:P(x(1)(x(1)(1)));"
        "6:(1);7:(1);8:P(x(1)(1));9:(1);10:P(1);11:P(1)"));
    ea.lbStore.destroy(lb);
}

TEST(typed_equality, find_digit_args_counts_only_digit_equality_arguments) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    gl::StrSpan out[gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS];

    // Row 84: interval has no input slots and `=` relates two sets, so the
    // theorem has no induction variable at all.
    const std::string row84(kRow84);
    ASSERT_EQ(ea.findDigitArgs(row84, ea.anchorInfo, out,
        gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS), 0);

    // A digit equality: in2's input 9 plus `=`'s inputs 9 and 10 (typed (1)),
    // minus in2's output 10 — the same answer as before the typing.
    const std::string digit =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(=[9,10])))";
    const int32_t n = ea.findDigitArgs(digit, ea.anchorInfo, out,
        gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS);
    ASSERT_EQ(n, 1);
    ASSERT_EQ(out[0].toStdString(), std::string("9"));

    // A lone `=` goal (the updateAdmissionMap3 call) types to the start
    // label, so both arguments still count.
    const std::string lone = "(=[10,11])";
    ASSERT_EQ(ea.findDigitArgs(lone, ea.anchorInfo, out,
        gl::ExecutionParameters::MAX_ADMISSION_REM_ARGS), 2);
}

namespace {

    /// The formal argument name a compact element uses at its last slot.
    std::string lastArgOf(const std::string& element) {
        gl::StrSpan args[gl::ExecutionParameters::MAX_ARITY];
        const int32_t n = gl::getArgsSpans(gl::StrSpan(element), args,
                                           gl::ExecutionParameters::MAX_ARITY);
        return args[n - 1].toStdString();
    }

    /// Checks that an extensionality entity's first element reads
    /// "u_1 subset of u_2" and its second "u_2 subset of u_1", through the
    /// compact's premise (subset side) and head (superset side).
    void assertSubsetOrientation(const gl::ExpressionAnalyzer& ea,
                                 const gl::LogicalEntity& e) {
        for (int k = 0; k < 2; ++k) {
            const std::string& instance = e.elements[static_cast<std::size_t>(k)];
            const gl::LogicalEntity* subset = ea.compiledEntity(
                gl::extractExpressionSpan(gl::StrSpan(instance)));
            ASSERT_TRUE(subset != nullptr);
            ASSERT_EQ(subset->category, std::string("implication"));
            ASSERT_EQ(subset->elements.size(), static_cast<std::size_t>(2));
            gl::StrSpan formals[gl::ExecutionParameters::MAX_ARITY];
            const int32_t formalN = gl::getArgsSpans(gl::StrSpan(subset->signature), formals,
                                                     gl::ExecutionParameters::MAX_ARITY);
            gl::StrSpan actuals[gl::ExecutionParameters::MAX_ARITY];
            gl::getArgsSpans(gl::StrSpan(instance), actuals, gl::ExecutionParameters::MAX_ARITY);
            const auto actualOf = [&](const std::string& formal) {
                for (int32_t i = 0; i < formalN; ++i)
                    if (formals[i].toStdString() == formal) return actuals[i].toStdString();
                return std::string("<none>");
            };
            const std::string premiseSet = actualOf(lastArgOf(subset->elements[0]));
            const std::string headSet = actualOf(lastArgOf(subset->elements[1]));
            ASSERT_EQ(premiseSet, std::string(k == 0 ? "u_1" : "u_2"));
            ASSERT_EQ(headSet, std::string(k == 0 ? "u_2" : "u_1"));
        }
    }

    /// The bare subproof payloads the goal door minted for one goal.
    std::vector<std::string> subproofPayloadsOf(gl::ExpressionAnalyzer& ea,
                                                gl::Memory& mb, const std::string& goal) {
        std::vector<std::string> out;
        for (gl::NameId id = 2; id <= mb.nameMap.nameCount(); ++id) {
            if (static_cast<std::size_t>(id) >= static_cast<std::size_t>(mb.nameMap.stackSize())
                || mb.nameMap.stackEmpty(id))
                continue;
            gl::StrSpan goalOut, bareOut;
            if (!ea.splitSubproofPayload(mb.nameMap.decodeSubView(mb.nameMap.stackBack(id)),
                                         goalOut, bareOut))
                continue;
            if (goalOut.toStdString() == goal) out.push_back(bareOut.toStdString());
        }
        return out;
    }

} // namespace

TEST(typed_equality, extensionality_entity_reuses_the_existing_subset_compact) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    const std::size_t before = ea.compiledExpressions.size();
    ea.ensureTypedEqualityEntity(gl::StrSpan(std::string("P(1)")));
    ea.ensureTypedEqualityEntity(gl::StrSpan(std::string("P(1)")));      // idempotent
    ASSERT_EQ(ea.typedEqualityEntities.size(), static_cast<std::size_t>(1));
    // FTA's registry already holds in[x,A] -> in[x,B]: nothing new is minted.
    ASSERT_EQ(ea.compiledExpressions.size(), before);

    const gl::LogicalEntity* e = ea.typedEqualityEntity(gl::StrSpan(std::string("P(1)")));
    ASSERT_TRUE(e != nullptr);
    ASSERT_EQ(e->category, std::string("and"));
    ASSERT_EQ(e->signature, std::string("(=[u_1,u_2])"));
    ASSERT_EQ(e->arity, 2);
    ASSERT_EQ(e->elements.size(), static_cast<std::size_t>(2));
    assertSubsetOrientation(ea, *e);
    const gl::LogicalEntity* subset = ea.compiledEntity(
        gl::extractExpressionSpan(gl::StrSpan(e->elements[0])));
    ASSERT_EQ(gl::extractExpressionSpan(gl::StrSpan(subset->elements[0])).toStdString(),
              std::string("in"));
    ASSERT_TRUE(ea.typedEqualityEntity(gl::StrSpan(std::string("P(x(1)(1))"))) == nullptr);
}

TEST(typed_equality, extensionality_entities_for_relations) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    const std::size_t before = ea.compiledExpressions.size();
    ea.ensureTypedEqualityEntity(gl::StrSpan(std::string("P(x(1)(1))")));
    ASSERT_EQ(ea.compiledExpressions.size(), before);       // reuses the in2 subset compact
    ea.ensureTypedEqualityEntity(gl::StrSpan(std::string("P(x(1)(x(1)(1)))")));
    ASSERT_EQ(ea.compiledExpressions.size(), before + 1);   // the in3 one is new

    const gl::LogicalEntity* binary = ea.typedEqualityEntity(gl::StrSpan(std::string("P(x(1)(1))")));
    const gl::LogicalEntity* ternary =
        ea.typedEqualityEntity(gl::StrSpan(std::string("P(x(1)(x(1)(1)))")));
    ASSERT_TRUE(binary != nullptr && ternary != nullptr);
    assertSubsetOrientation(ea, *binary);
    assertSubsetOrientation(ea, *ternary);
    const gl::LogicalEntity* ternarySubset = ea.compiledEntity(
        gl::extractExpressionSpan(gl::StrSpan(ternary->elements[0])));
    ASSERT_EQ(gl::extractExpressionSpan(gl::StrSpan(ternarySubset->elements[0])).toStdString(),
              std::string("in3"));
}

TEST(typed_equality, theorem_registration_compiles_only_what_its_head_needs) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    const std::vector<std::string> digit = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])", "(in2[9,10,3])", "(=[9,10])" };
    gl::StrSpan spans[4];
    for (int i = 0; i < 3; ++i) spans[i] = gl::StrSpan(digit[i]);
    ea.ensureTypedEqualityEntityForTheorem(spans, 3);
    ASSERT_TRUE(ea.typedEqualityEntities.empty());        // a digit head needs nothing

    const std::vector<std::string> sets = {
        "(AnchorFTA[1,2,3,4,5,6,7,8])", "(interval[1,4,2,9,10])",
        "(interval[1,4,2,9,11])", "(=[10,11])" };
    for (int i = 0; i < 4; ++i) spans[i] = gl::StrSpan(sets[i]);
    ea.ensureTypedEqualityEntityForTheorem(spans, 4);
    ASSERT_EQ(ea.typedEqualityEntities.size(), static_cast<std::size_t>(1));
    ASSERT_TRUE(ea.typedEqualityEntity(gl::StrSpan(std::string("P(1)"))) != nullptr);
}

TEST(typed_equality, goal_label_from_the_lb_records) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.addTheoremToMemory(std::string(kRow84), ea.body, 0, false, ea.globalDependencies);
    gl::Memory* secondLB = childAt(ea, { "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(interval[1,4,2,9,10])", "(interval[1,4,2,9,11])" });
    ASSERT_TRUE(secondLB != nullptr);
    ASSERT_EQ(ea.resolveEqualityGoalLabel(gl::StrSpan(std::string("(=[10,11])")), *secondLB,
                                          gl::StrSpan("main", 4)).toStdString(),
              std::string("P(1)"));
    // One recorded side types both.
    ASSERT_EQ(ea.resolveEqualityGoalLabel(gl::StrSpan(std::string("(=[10,zz])")), *secondLB,
                                          gl::StrSpan("main", 4)).toStdString(),
              std::string("P(1)"));
    ASSERT_EQ(ea.resolveEqualityGoalLabel(gl::StrSpan(std::string("(=[9,2])")), *secondLB,
                                          gl::StrSpan("main", 4)).toStdString(),
              std::string("(1)"));
    ASSERT_EQ(gl::ExpressionAnalyzer::chainVarLabel(*secondLB, gl::StrSpan("zz", 2)).toStdString(),
              std::string(""));
}

TEST(typed_equality, goal_label_from_the_subproof_element) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.addTheoremToMemory(std::string(kRow84), ea.body, 0, false, ea.globalDependencies);
    gl::Memory* lb = childAt(ea, { "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(interval[1,4,2,9,10])", "(interval[1,4,2,9,11])" });
    ASSERT_TRUE(lb != nullptr);
    // An element whose `=` head is typed inside it: two intervals, equal.
    ea.compiledExpressions.emplace("typedEqTestIntervalEq", gl::LogicalEntity(
        "implication", { "(interval[u_1,u_2,u_3,u_4,1])", "(interval[u_1,u_2,u_3,u_4,2])",
                         "(=[1,2])" },
        "(typedEqTestIntervalEq[u_1,u_2,u_3,u_4])", 4));
    // An element whose `=` head is tied to a formal only: typed through the
    // formal's actual argument in the LB records.
    ea.compiledExpressions.emplace("typedEqTestFormal", gl::LogicalEntity(
        "implication", { "(in[5,u_2])", "(=[1,u_1])" }, "(typedEqTestFormal[u_1,u_2])", 2));

    const auto scopeFor = [&](const std::string& payload) {
        const gl::NameId mainId = lb->nameMap.encode(std::string("main"));
        const gl::NameId v = lb->nameMap.encodePush(mainId, gl::StrSpan(payload));
        return lb->nameMap.decode(v);
    };
    const std::string intervalScope = scopeFor("(typedEqTestIntervalEq[1,4,2,9])");
    ASSERT_EQ(ea.resolveEqualityGoalLabel(gl::StrSpan(std::string("(=[repl_lev_3_0,repl_lev_3_1])")),
                                          *lb, gl::StrSpan(intervalScope)).toStdString(),
              std::string("P(1)"));
    const std::string formalScope = scopeFor("(=[10,11])_subproof_(typedEqTestFormal[10,1])");
    ASSERT_EQ(ea.resolveEqualityGoalLabel(gl::StrSpan(std::string("(=[repl_lev_3_2,repl_lev_3_3])")),
                                          *lb, gl::StrSpan(formalScope)).toStdString(),
              std::string("P(1)"));
    // A head group of bound variables only resolves inside the element, to
    // `=`'s starting point.
    ea.compiledExpressions.emplace("typedEqTestBound", gl::LogicalEntity(
        "implication", { "(in[5,u_1])", "(=[6,7])" }, "(typedEqTestBound[u_1])", 1));
    const std::string boundScope = scopeFor("(typedEqTestBound[10])");
    ASSERT_EQ(ea.resolveEqualityGoalLabel(gl::StrSpan(std::string("(=[repl_lev_3_4,repl_lev_3_5])")),
                                          *lb, gl::StrSpan(boundScope)).toStdString(),
              std::string("(1)"));
    // At main nothing but the records types a variable.
    const gl::NameId mainId = lb->nameMap.encode(std::string("main"));
    ASSERT_TRUE(ea.scopeVarLabel(gl::StrSpan("zz", 2), *lb, mainId).empty());
    ASSERT_EQ(ea.scopeVarLabel(gl::StrSpan("10", 2), *lb, mainId).toStdString(),
              std::string("P(1)"));
}

TEST(typed_equality, goal_label_from_the_or_element_and_parent_scopes) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.addTheoremToMemory(std::string(kRow84), ea.body, 0, false, ea.globalDependencies);
    gl::Memory* lb = childAt(ea, { "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(interval[1,4,2,9,10])", "(interval[1,4,2,9,11])" });
    ASSERT_TRUE(lb != nullptr);
    const gl::NameId mainId = lb->nameMap.encode(std::string("main"));
    const auto push = [&](gl::NameId parent, const std::string& payload) {
        return lb->nameMap.encodePush(parent, gl::StrSpan(payload));
    };
    const auto goalLabel = [&](const std::string& goal, gl::NameId scope) {
        return ea.resolveEqualityGoalLabel(gl::StrSpan(goal), *lb,
            gl::StrSpan(lb->nameMap.decode(scope))).toStdString();
    };

    // OR-case branches: an actual is typed through the OR element's slot for
    // it, or through a joined formal's actual in the LB records.
    ea.compiledExpressions.emplace("typedEqTestOrSlot", gl::LogicalEntity(
        "or", { "(=[u_1,u_2])", "(in[u_3,u_1])" }, "(typedEqTestOrSlot[u_1,u_2,u_3])", 3));
    ea.compiledExpressions.emplace("typedEqTestOrFree", gl::LogicalEntity(
        "or", { "(=[u_1,u_2])", "(=[u_2,u_3])" }, "(typedEqTestOrFree[u_1,u_2,u_3])", 3));
    const gl::NameId slotScope = push(mainId, "orint_(typedEqTestOrSlot[aa,bb,cc])_((=[aa,bb]))");
    ASSERT_EQ(goalLabel("(=[aa,bb])", slotScope), std::string("P(1)"));
    const gl::NameId freeScope = push(mainId, "orint_(typedEqTestOrFree[10,dd,ee])_((=[dd,ee]))");
    ASSERT_EQ(goalLabel("(=[dd,ee])", freeScope), std::string("P(1)"));
    // A disjunct argument that is no actual of the OR has no answer there.
    ASSERT_TRUE(ea.scopeGoalArgLabel(gl::StrSpan(std::string("(=[dd,qq])")), 1, *lb,
                                     freeScope).empty());

    // Nested subproofs: the inner head is tied to its formal only; the
    // formal's actual is a bound variable of the outer subproof, typed
    // through the outer goal and the outer element's slots.
    ea.compiledExpressions.emplace("typedEqTestInner", gl::LogicalEntity(
        "implication", { "(=[3,u_1])", "(=[4,3])" }, "(typedEqTestInner[u_1])", 1));
    ea.compiledExpressions.emplace("typedEqTestOuter", gl::LogicalEntity(
        "implication", { "(interval[u_1,u_2,u_3,u_4,1])", "(typedEqTestInner[1])" },
        "(typedEqTestOuter[u_1,u_2,u_3,u_4])", 4));
    const gl::NameId outer = push(mainId, "(=[10,11])_subproof_(typedEqTestOuter[1,4,2,9])");
    ASSERT_TRUE(gl::ExpressionAnalyzer::goalAtScope(*lb, outer).empty());
    const std::string outerGoal = "(typedEqTestInner[repl_lev_3_5])";
    lb->intToBeProved.assignSet(gl::packStatementKey(
        lb->nameMap.encode(outerGoal), outer), nullptr, 0);
    ASSERT_EQ(gl::ExpressionAnalyzer::goalAtScope(*lb, outer).toStdString(), outerGoal);
    const gl::NameId inner = push(outer, outerGoal);
    ASSERT_EQ(goalLabel("(=[repl_lev_3_6,repl_lev_3_7])", inner), std::string("P(1)"));
    ASSERT_EQ(ea.scopeVarLabel(gl::StrSpan(std::string("repl_lev_3_5")), *lb, outer).toStdString(),
              std::string("P(1)"));
    ASSERT_EQ(ea.elementVarLabel(gl::StrSpan(outerGoal), gl::StrSpan("4", 1), *lb, outer)
                  .toStdString(), std::string("P(1)"));
    // A scope no element opened passes the question to its parent.
    const gl::NameId branch = push(outer, "ordis_(typedEqTestOrFree[1,2,3])_((=[1,2]))");
    ASSERT_EQ(ea.scopeVarLabel(gl::StrSpan(std::string("repl_lev_3_5")), *lb, branch).toStdString(),
              std::string("P(1)"));
    // An actual nothing types leaves the group untyped.
    ASSERT_TRUE(ea.elementVarLabel(gl::StrSpan(std::string("(typedEqTestInner[zz])")),
                                   gl::StrSpan("4", 1), *lb, mainId).empty());
}

TEST(typed_equality, set_goal_opens_two_subset_subproofs_digit_goal_stays_atomic) {
    gl::ExpressionAnalyzer ea(std::string("FTA"));
    ea.addTheoremToMemory(std::string(kRow84), ea.body, 0, false, ea.globalDependencies);
    gl::Memory* setLB = childAt(ea, { "(AnchorFTA[1,2,3,4,5,6,7,8])",
        "(interval[1,4,2,9,10])", "(interval[1,4,2,9,11])" });
    ASSERT_TRUE(setLB != nullptr);
    const gl::LogicalEntity* e = ea.typedEqualityEntity(gl::StrSpan(std::string("P(1)")));
    ASSERT_TRUE(e != nullptr);
    const std::string compact =
        gl::extractExpressionSpan(gl::StrSpan(e->elements[0])).toStdString();

    std::vector<std::string> payloads = subproofPayloadsOf(ea, *setLB, "(=[10,11])");
    std::sort(payloads.begin(), payloads.end());
    std::vector<std::string> expected = {
        "(" + compact + "[10,11])", "(" + compact + "[11,10])" };
    std::sort(expected.begin(), expected.end());
    ASSERT_TRUE(payloads == expected);

    const std::string digit =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10](in2[9,10,3])(=[9,10])))";
    ea.addTheoremToMemory(digit, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* digitLB = childAt(ea, { "(AnchorFTA[1,2,3,4,5,6,7,8])", "(in2[9,10,3])" });
    ASSERT_TRUE(digitLB != nullptr);
    ASSERT_TRUE(subproofPayloadsOf(ea, *digitLB, "(=[9,10])").empty());
}
