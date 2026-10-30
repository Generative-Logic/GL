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
/// @brief Unit tests for the int scopes of premise-bound integrations
///        (D-578).
///
/// @details
/// When a weak induction hypothesis install files identical integration
/// goals once per premise, the first fill of such a premise-bound copy opens
/// its own scope `<parent>_boundary_int_scope_<premise>_<witness>`, where the
/// witness's subtree is published and the copy continues. A first fill
/// accepts only an arriving expression whose free arguments are numbers or
/// `rec`, through a key that holds `rec` or the induction variable. Per
/// premise the first scope to finish (an instance of its integration goal
/// registered there) counts; once every premise of the group has one, those
/// scopes are promoted to their parents together and then wiped. Covers the
/// scope syntax, the gates, the opening on the route-3 commit and on the
/// statement fill, the in-place continuation, the finish, the promotion and
/// the origin tag.

#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "../logical_block/derivation_tree.hpp"
#include "../prover.hpp"

#include <cstring>
#include <string>
#include <vector>

namespace {
    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief The tests' page-tier scratch arena.
    gl::ScratchArena& pages() {
        return gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    }

    /// @brief The tests' string-tier scratch arena.
    gl::ScratchArena& strings() {
        return gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    }

    /// @brief The split binding of a copy (tests only).
    gl::IntegrationForbiddenSets binding(uint64_t variables, uint64_t levels, int32_t premiseLevel) {
        gl::IntegrationForbiddenSets out;
        out.variables = variables;
        out.levels = levels;
        out.premiseLevel = premiseLevel;
        return out;
    }

    /// @brief Row 20's copy bound to P (level 1).
    const gl::IntegrationForbiddenSets kCopyP = binding(uint64_t{1} << 12, uint64_t{1} << 2, 1);
    /// @brief Row 20's copy bound to Q (level 2).
    const gl::IntegrationForbiddenSets kCopyQ = binding(uint64_t{1} << 10, uint64_t{1} << 1, 2);

    const char kPremiseP[] = "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])";
    const char kPremiseQ[] = "(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])";

    /// @brief A weak step block (the fixture's body, level 3) below the
    ///        anchor block and the blocks of row 20's premises P (level 1)
    ///        and Q (level 2), with its recursion hypothesis
    ///        `(in2[rec,9,3])`, `main` registered and its canonical scope
    ///        initialized; its hypothesis install split identical goals.
    struct IntScopeFixture {
        gl::tests::ProverFixture analyzer{"Peano"};
        gl::Memory root;
        gl::Memory anchor;
        gl::Memory premiseP;
        gl::Memory premiseQ;
        gl::Memory& memory;

        IntScopeFixture() : memory(analyzer.body) {
            analyzer.anchorInfo.name = "AnchorTest92";
            analyzer.parameters.trackHistory = true;
            anchor.parentMemory = &root;
            premiseP.parentMemory = &anchor;
            premiseQ.parentMemory = &premiseP;
            memory.parentMemory = &premiseQ;
            anchor.setExprKey("(AnchorTest92[N,0,s])");
            premiseP.setExprKey(kPremiseP);
            premiseQ.setExprKey(kPremiseQ);
            premiseP.level = 1;
            premiseQ.level = 2;
            memory.level = 3;
            memory.isActive = true;
            memory.nameMap.encode(sp("main"));
            memory.recursionHypothesisId = memory.nameMap.encode(sp("(in2[rec,9,3])"));
            memory.inductionZeroId = memory.nameMap.encode(sp("2"));
            memory.recursionAuxyImplId = 1;
            memory.premiseSplitFiled = true;
            analyzer.canonicalizer.initializeCanonicalScope(memory, gl::NameMap::MAIN_ID);
        }
    };

    /// @brief File under @p key at @p validity one copy per binding of the
    ///        instruction `(in2[u_2,marker,u_3])` then @p second (by default
    ///        `(in3[u_2,u_3,pi_x,u_4])`, whose continuation keys
    ///        `(in3[u_2,u_3,marker,u_4])`).
    void fileTwoPartCopies(gl::Memory& m, const std::string& key, const std::string& validity,
                           const std::vector<gl::IntegrationForbiddenSets>& copies,
                           const std::string& second = "(in3[u_2,u_3,pi_x,u_4])") {
        gl::ScratchArena& arena = pages();
        gl::ScratchScope frame(arena);
        gl::WorkInstruction wi(&arena);
        int32_t mark = wi.elemMark();
        wi.commitEntity(sp("atomic"), sp("(in2[u_2,marker,u_3])"), sp(""), 3, mark);
        mark = wi.elemMark();
        wi.commitEntity(sp("atomic"), gl::StrSpan(second), sp(""), 4, mark);
        wi.setMarkedGoal(sp("(goal74[1,2,3])"));
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, validity);
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        for (const gl::IntegrationForbiddenSets& copy : copies) {
            wi.forbidden = copy;
            entries.findOrEmplaceWork(wi, m.valueInterner);
        }
        entries.writeToCold(m.overallHashMemory.admissionMapIntegration, pk);
        m.lbMemory.canonicalUses.integrationDemands.changed(pk);
    }

    /// @brief File under @p key at @p validity one unbound instruction
    ///        @p first (by default `(in2[u_2,marker,u_3])`) then @p second,
    ///        with the marked goal @p goal; a fill of the first part keys the
    ///        second part's open argument in the continuation.
    void fileInstruction(gl::Memory& m, const std::string& key, const std::string& validity,
                         const std::string& second, const std::string& first = "(in2[u_2,marker,u_3])",
                         const std::string& goal = "(goal74[1,2,3])") {
        gl::ScratchArena& arena = pages();
        gl::ScratchScope frame(arena);
        gl::WorkInstruction wi(&arena);
        int32_t mark = wi.elemMark();
        wi.commitEntity(sp("atomic"), gl::StrSpan(first), sp(""), 3, mark);
        mark = wi.elemMark();
        wi.commitEntity(sp("atomic"), gl::StrSpan(second), sp(""), 4, mark);
        wi.setMarkedGoal(gl::StrSpan(goal));
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, validity);
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        entries.findOrEmplaceWork(wi, m.valueInterner);
        entries.writeToCold(m.overallHashMemory.admissionMapIntegration, pk);
        m.lbMemory.canonicalUses.integrationDemands.changed(pk);
    }

    /// @brief Whether the admission map holds instructions under @p key at
    ///        @p validity.
    bool keyFiled(gl::Memory& m, const std::string& key, const std::string& validity) {
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, validity);
        const int32_t id = m.overallHashMemory.admissionMapIntegration.lookup(pk);
        return id != 0 && m.overallHashMemory.admissionMapIntegration.runLen(id) > 0;
    }

    /// @brief The payload values of the copy bound to @p premiseLevel under
    ///        @p key at @p validity.
    std::vector<std::string> copyValues(gl::Memory& m, const std::string& key, const std::string& validity,
                                        int32_t premiseLevel) {
        gl::ScratchArena& arena = pages();
        gl::ScratchScope frame(arena);
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, validity);
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        std::vector<std::string> values;
        for (int32_t i = 0; i < entries.entries.size(); ++i) {
            if (entries.entries[i].forbidden.premiseLevel != premiseLevel) continue;
            int32_t ids[gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES];
            const int32_t n = entries.valuesAt(i, ids, gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES);
            for (int32_t v = 0; v < n; ++v) values.push_back(m.valueInterner.decode(ids[v]));
        }
        return values;
    }

    /// @brief An arriving-expression probe whose start is @p arriving.
    gl::WitnessHistoryProbe arrival(const char* arriving) {
        gl::WitnessHistoryProbe probe;
        probe.start = sp(arriving);
        probe.startScope = sp("main");
        return probe;
    }

    /// @brief One statement through the statement door at @p scope with
    ///        level @p level.
    void addAt(gl::tests::ProverFixture& analyzer, const std::string& statement, const std::string& scope,
               int level, gl::PagedVector<gl::IntEncodedExpr>& out) {
        const int levels[1] = { level };
        const gl::TransientOrigin origin{ true, gl::OriginTag::broadcast, nullptr, 0 };
        analyzer.statementMemory.addStatement(gl::StrSpan(statement), analyzer.body, true, levels, 1,
            origin, gl::StrSpan(scope), out);
    }

    /// @brief Whether @p statement is known at @p scope.
    bool knownAt(gl::Memory& m, const std::string& statement, const std::string& scope) {
        return gl::lookupStatementFlags(m.intKnownStatements, m.nameMap,
            gl::StrSpan(statement), gl::StrSpan(scope)) != nullptr;
    }

    /// @brief A premise-bound instruction with the marked goal @p goal.
    void boundInstruction(gl::WorkInstruction& wi, const char* goal,
                          const gl::IntegrationForbiddenSets& copy) {
        const int32_t mark = wi.elemMark();
        wi.commitEntity(sp("atomic"), sp("(in2[u_2,marker,u_3])"), sp(""), 3, mark);
        wi.setMarkedGoal(sp(goal));
        wi.forbidden = copy;
    }
} // namespace

// The payload is `int_scope_<premise>_<witness>`; the kind test is a prefix
// test on the payload, which no induction, trial, subproof or witness-name
// payload passes.
TEST(int_scope, payload_kind_and_builder) {
    ASSERT_TRUE(gl::ScopeSyntax::isIntScopePayload(sp("int_scope_(p[1,9])_w")));
    ASSERT_FALSE(gl::ScopeSyntax::isIntScopePayload(sp("int_scope_")));
    ASSERT_FALSE(gl::ScopeSyntax::isIntScopePayload(sp("is_(p[marker])_w")));
    ASSERT_FALSE(gl::ScopeSyntax::isIntScopePayload(sp("integration_int_0_lev_6_119")));
    ASSERT_FALSE(gl::ScopeSyntax::isIntScopePayload(sp("int_0_lev_3_0")));
    ASSERT_FALSE(gl::ScopeSyntax::isIntScopePayload(sp("(implication3[int_scope_x])")));

    gl::ScratchScope frame(strings());
    const gl::ScratchString payload = gl::ScopeSyntax::intScopePayloadScratch(strings(),
        sp(kPremiseP), sp("int_0_lev_3_0"));
    ASSERT_EQ(gl::StrSpan(payload).toStdString(),
        std::string("int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])_int_0_lev_3_0"));

    gl::Memory m;
    gl::NameMap& names = m.nameMap;
    names.encode(sp("main"));
    const gl::NameId intScope = names.encodePush(gl::NameMap::MAIN_ID, gl::StrSpan(payload));
    const gl::NameId nested = names.encodePush(intScope, std::string("(implication3[1,2])"));
    const gl::NameId induction = names.encodePush(gl::NameMap::MAIN_ID,
        std::string("is_(in3[2,3,marker,4])_it_0_lev_2_7"));
    ASSERT_TRUE(gl::ScopeSyntax::isIntScope(names, intScope));
    ASSERT_FALSE(gl::ScopeSyntax::isIntScope(names, nested));
    ASSERT_FALSE(gl::ScopeSyntax::isIntScope(names, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gl::ScopeSyntax::isIntScope(names, induction));
    ASSERT_TRUE(gl::ScopeSyntax::atOrBelowIntScope(names, nested));
    ASSERT_FALSE(gl::ScopeSyntax::atOrBelowIntScope(names, induction));
    ASSERT_FALSE(gl::ScopeSyntax::canHoldGoal(names, intScope));
}

// A statement is an instance of the goal when its core, sign and arity match
// and every argument that is not a marker is equal.
TEST(int_scope, goal_instance_match) {
    const gl::StrSpan goal = sp("(primeFactorSeq[1,3,4,5,2,6,7,rec,marker,marker])");
    ASSERT_TRUE(gl::ScopeSyntax::intScopeGoalMatches(goal,
        sp("(primeFactorSeq[1,3,4,5,2,6,7,rec,int_0_lev_3_0,int_1_lev_3_132])")));
    ASSERT_FALSE(gl::ScopeSyntax::intScopeGoalMatches(goal,
        sp("(primeFactorSeq[1,3,4,5,2,6,7,9,int_0_lev_3_0,int_1_lev_3_132])")));
    ASSERT_FALSE(gl::ScopeSyntax::intScopeGoalMatches(goal,
        sp("!(primeFactorSeq[1,3,4,5,2,6,7,rec,int_0_lev_3_0,int_1_lev_3_132])")));
    ASSERT_FALSE(gl::ScopeSyntax::intScopeGoalMatches(goal, sp("(fold[1,3,4,5,a,2,rec,b])")));
    ASSERT_FALSE(gl::ScopeSyntax::intScopeGoalMatches(goal, sp("(primeFactorSeq[1,3,4,5,2,6,7,rec,a])")));
    ASSERT_TRUE(gl::ScopeSyntax::intScopeGoalMatches(sp("(in3[2,marker1,marker2,4])"), sp("(in3[2,a,b,4])")));
}

// A split preparation's first keys must hold `rec` or the step block's
// induction variable.
TEST(int_scope, first_keys_hold_rec_or_the_induction_variable) {
    IntScopeFixture f;
    ASSERT_TRUE(gl::ScopeSyntax::keyHoldsInductionVariable(f.memory, sp("(sequence[u_1,u_4,u_2,u_rec,marker])")));
    ASSERT_TRUE(gl::ScopeSyntax::keyHoldsInductionVariable(f.memory, sp("(fold[u_1,u_9,marker])")));
    ASSERT_FALSE(gl::ScopeSyntax::keyHoldsInductionVariable(f.memory, sp("(ordered[u_1,u_4,marker])")));
    ASSERT_FALSE(gl::ScopeSyntax::keyHoldsInductionVariable(f.memory, sp("(fold[u_1,u_19,marker,u_recx])")));
}

// The initiation gate reads the free arguments: a binder's own names are
// left out, everything else must be a number or `rec`.
TEST(int_scope, free_arguments_are_numbers) {
    ASSERT_TRUE(gl::hasOnlyNumberFreeArguments(sp("(existence12[1,4,rec,10,2])")));
    ASSERT_TRUE(gl::hasOnlyNumberFreeArguments(sp("(p[u_1,u_rec])")));
    ASSERT_FALSE(gl::hasOnlyNumberFreeArguments(sp("(existence12[1,4,rec,force_lev_1_2,2])")));
    ASSERT_FALSE(gl::hasOnlyNumberFreeArguments(sp("(existence12[1,4,rec,int_0_lev_3_0,2])")));
    ASSERT_TRUE(gl::hasOnlyNumberFreeArguments(sp("!(>[I](interval[1,4,2,rec,I])!(fXY[10,I,1]))")));
    ASSERT_FALSE(gl::hasOnlyNumberFreeArguments(sp("!(>[I](interval[1,4,2,rec,I])!(fXY[R,I,1]))")));
}

// An unbound instruction never refuses; a premise-bound copy refuses its
// forbidden members everywhere and, outside every int scope, an arriving
// expression with an argument that is not a number.
TEST(int_scope, premise_copy_refusal) {
    gl::Memory m;
    gl::NameMap& names = m.nameMap;
    names.encode(sp("main"));
    const gl::NameId intScope = names.encodePush(gl::NameMap::MAIN_ID, std::string("int_scope_(p[1,9])_w"));
    const gl::IntegrationForbiddenSets unbound;
    ASSERT_FALSE(gl::premiseCopyRefuses(m, gl::NameMap::MAIN_ID, unbound,
        sp("(existence12[1,4,rec,force_lev_1_2,12])")));
    ASSERT_TRUE(gl::premiseCopyRefuses(m, gl::NameMap::MAIN_ID, kCopyP, sp("(existence12[1,4,rec,12,2])")));
    ASSERT_FALSE(gl::premiseCopyRefuses(m, gl::NameMap::MAIN_ID, kCopyQ, sp("(existence12[1,4,rec,12,2])")));
    ASSERT_TRUE(gl::premiseCopyRefuses(m, gl::NameMap::MAIN_ID, kCopyQ,
        sp("(existence12[1,4,rec,int_0_lev_3_0,2])")));
    ASSERT_FALSE(gl::premiseCopyRefuses(m, intScope, kCopyQ, sp("(existence12[1,4,rec,int_0_lev_3_0,2])")));
    ASSERT_TRUE(gl::premiseCopyRefuses(m, intScope, kCopyQ, sp("(existence12[1,4,rec,10,2])")));
    ASSERT_TRUE(gl::premiseCopyRefuses(m, intScope, kCopyP, sp("(existence12[1,4,rec,force_lev_2_2,2])")));
}

// The promotion row's tag is the raw-only `int scope`, last in the table.
TEST(int_scope, origin_tag_name) {
    ASSERT_EQ(std::string(gl::originTagName(gl::OriginTag::intScope)), std::string("int scope"));
    ASSERT_TRUE(gl::originTagFromString(std::string("int scope")) == gl::OriginTag::intScope);
    ASSERT_EQ(static_cast<int>(gl::OriginTag::intScope) + 1, static_cast<int>(gl::OriginTag::COUNT));
}

// A route-3 commit that first fills a premise-bound copy at `main` opens the
// copy's int scope named by its premise and the witness, records the scope
// and its group, and continues the copy there: the continuation key is
// filed at the int scope, not at `main`. The copy bound to the other premise
// refuses the arrival.
TEST(int_scope, commit_opens_the_premise_scope_and_continues_there) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in2[u_2,marker,u_3])";
    fileTwoPartCopies(f.memory, key, "main", { kCopyP, kCopyQ });
    const gl::WitnessHistoryProbe fromQ = arrival("(existence12[1,4,rec,12,2])");
    gl::InductionScopeSink opened;
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
        sp("main"), true, 0, nullptr, 0, &fromQ, &opened));
    ASSERT_EQ(opened.count, 1);
    ASSERT_FALSE(opened.publishedHere);
    const std::string scope = "main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])_w";
    ASSERT_EQ(f.memory.nameMap.decode(opened.scopes[0]), scope);
    ASSERT_TRUE(copyValues(f.memory, key, "main", 2) == std::vector<std::string>{ "w" });
    ASSERT_TRUE(copyValues(f.memory, key, "main", 1).empty());
    ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_4])", scope));
    ASSERT_FALSE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_4])", "main"));

    const int64_t* record = f.memory.lbMemory.intScopes.find(static_cast<int64_t>(opened.scopes[0]));
    ASSERT_TRUE(record != nullptr);
    const gl::NameId group = static_cast<gl::NameId>(static_cast<int32_t>(static_cast<uint64_t>(*record) >> 32));
    ASSERT_EQ(static_cast<int32_t>(static_cast<uint32_t>(*record)), 2);
    ASSERT_EQ(f.memory.nameMap.decode(group), std::string("(goal74[1,2,3])"));
    const int64_t* levels = f.memory.lbMemory.intScopeGroups.find(static_cast<int64_t>(group));
    ASSERT_TRUE(levels != nullptr);
    ASSERT_EQ(*levels, int64_t{(1 << 1) | (1 << 2)});
}

// Inside an int scope a commit of the scope's own copy continues in place:
// no further int scope opens, the initiation gate is off, and the witness is
// published where the commit runs.
TEST(int_scope, commit_inside_an_int_scope_continues_in_place) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string scope = f.memory.nameMap.decode(f.memory.nameMap.encodePush(gl::NameMap::MAIN_ID,
        std::string("int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])_w")));
    f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, f.memory.nameMap.lookup(gl::StrSpan(scope)));
    const std::string key = "(in2[u_2,marker,u_3])";
    fileTwoPartCopies(f.memory, key, scope, { kCopyQ });
    gl::WitnessHistoryProbe forced = arrival("(existence12[1,4,rec,force_lev_2_2,2])");
    forced.startScope = gl::StrSpan(scope);
    gl::InductionScopeSink opened;
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,v,3])"), sp("v"), sp("(in2[2,marker,3])"),
        gl::StrSpan(scope), true, 0, nullptr, 0, &forced, &opened));
    ASSERT_EQ(opened.count, 0);
    ASSERT_TRUE(opened.publishedHere);
    ASSERT_TRUE(copyValues(f.memory, key, scope, 2) == std::vector<std::string>{ "v" });
    ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_4])", scope));
    ASSERT_EQ(f.memory.lbMemory.intScopes.count(), 0);
}

// An instruction without a premise binding admits at the offering scope as
// before: no int scope opens.
TEST(int_scope, unbound_instruction_admits_at_the_offering_scope) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in2[u_2,marker,u_3])";
    fileTwoPartCopies(f.memory, key, "main", { gl::IntegrationForbiddenSets{} });
    const gl::WitnessHistoryProbe any = arrival("(existence12[1,4,rec,int_0_lev_3_0,2])");
    gl::InductionScopeSink opened;
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
        sp("main"), true, 0, nullptr, 0, &any, &opened));
    ASSERT_EQ(opened.count, 0);
    ASSERT_TRUE(opened.publishedHere);
    ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_4])", "main"));
    ASSERT_EQ(f.memory.lbMemory.intScopes.count(), 0);
}

// One premise and witness open one scope; a closed scope never reopens.
TEST(int_scope, closed_scope_never_reopens) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    gl::WorkInstruction wi(&arena);
    boundInstruction(wi, "(goal74[1,marker1,marker2])", kCopyP);
    const gl::NameId first = gate.openIntScope(f.memory, gl::NameMap::MAIN_ID, wi, sp("r"));
    ASSERT_TRUE(first != 0);
    ASSERT_EQ(f.memory.nameMap.decode(first),
        std::string("main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])_r"));
    ASSERT_EQ(gate.openIntScope(f.memory, gl::NameMap::MAIN_ID, wi, sp("r")), first);
    ASSERT_EQ(f.memory.lbMemory.intScopes.count(), 1);
    const gl::NameId group = f.memory.nameMap.lookup(gl::StrSpan(std::string("(goal74[1,marker,marker])")));
    ASSERT_TRUE(group != 0);
    f.memory.lbMemory.intValidityNamesToFilter.mint(first);
    ASSERT_EQ(gate.openIntScope(f.memory, gl::NameMap::MAIN_ID, wi, sp("r")), gl::NameId(0));
}

// The statement fill at `main` opens a copy's int scope from an arrival of
// numbers; at an int scope it fills the scope's own keys in place; elsewhere
// it fills nothing.
TEST(int_scope, statement_fill_opens_and_fills_int_scopes) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in2[u_2,marker,u_3])";
    fileTwoPartCopies(f.memory, key, "main", { kCopyP, kCopyQ });
    gate.updateAdmissionMapIntegration(sp("(in2[2,t,3])"), f.memory, sp("main"), 0, 0, true);
    const std::string scopeP = "main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])_t";
    const std::string scopeQ = "main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])_t";
    ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_4])", scopeP));
    ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_4])", scopeQ));
    ASSERT_FALSE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_4])", "main"));

    fileTwoPartCopies(f.memory, key, scopeQ, { kCopyQ });
    gate.updateAdmissionMapIntegration(sp("(in2[2,int_0_lev_3_5,3])"), f.memory, gl::StrSpan(scopeQ), 0, 0, false);
    ASSERT_TRUE(copyValues(f.memory, key, scopeQ, 2) == std::vector<std::string>{ "int_0_lev_3_5" });

    const std::string subproof = f.memory.nameMap.decode(f.memory.nameMap.encodePush(gl::NameMap::MAIN_ID,
        std::string("(implication3[1,2])")));
    fileTwoPartCopies(f.memory, key, subproof, { kCopyQ });
    gate.updateAdmissionMapIntegration(sp("(in2[2,u,3])"), f.memory, gl::StrSpan(subproof), 0, 0, true);
    ASSERT_TRUE(copyValues(f.memory, key, subproof, 2).empty());
}

// Per premise the first scope to finish counts; when every premise of the
// group has finished, those scopes are promoted to `main` together with an
// `int scope` row citing each statement at its scope and keeping its levels;
// a scope that is not a first finisher is never promoted. The promotion
// closes the group: every int scope of the group, promoted or not, is queued
// for wiping and wiped, the group's records stay, and the group opens no
// further int scope (D-582).
TEST(int_scope, first_finished_scopes_are_promoted_together) {
    IntScopeFixture f;
    gl::Memory& memory = f.memory;
    auto& gate = f.analyzer.admissionMemory;
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    gl::WorkInstruction wiP(&arena);
    gl::WorkInstruction wiQ(&arena);
    boundInstruction(wiP, "(in3[2,marker1,marker2,4])", kCopyP);
    boundInstruction(wiQ, "(in3[2,marker2,marker1,4])", kCopyQ);
    const gl::NameId p = gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiP, sp("r"));
    const gl::NameId p2 = gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiP, sp("r2"));
    const gl::NameId q = gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiQ, sp("s"));
    const std::string scopeP = memory.nameMap.decode(p);
    const std::string scopeP2 = memory.nameMap.decode(p2);
    const std::string scopeQ = memory.nameMap.decode(q);
    const gl::NameId group = memory.nameMap.lookup(gl::StrSpan(std::string("(in3[2,marker,marker,4])")));
    ASSERT_TRUE(group != 0);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);

    addAt(f.analyzer, "(in2[2,r,3])", scopeP, 3, out);
    addAt(f.analyzer, "(in3[2,r,m,4])", scopeP, 4, out);
    const int64_t* firstP = memory.lbMemory.intScopeFirstFinish.find(gl::packInt32Pair(group, 1));
    ASSERT_TRUE(firstP != nullptr && *firstP == static_cast<int64_t>(p));
    ASSERT_FALSE(knownAt(memory, "(in3[2,r,m,4])", "main"));
    addAt(f.analyzer, "(in3[2,r2,n,4])", scopeP2, 4, out);
    ASSERT_EQ(*memory.lbMemory.intScopeFirstFinish.find(gl::packInt32Pair(group, 1)), static_cast<int64_t>(p));
    ASSERT_EQ(memory.pendingWipeScopes.count(), 0);

    addAt(f.analyzer, "(in2[2,s,3])", scopeQ, 3, out);
    addAt(f.analyzer, "(in3[2,s,m,4])", scopeQ, 5, out);
    ASSERT_TRUE(knownAt(memory, "(in2[2,r,3])", "main"));
    ASSERT_TRUE(knownAt(memory, "(in3[2,r,m,4])", "main"));
    ASSERT_TRUE(knownAt(memory, "(in2[2,s,3])", "main"));
    ASSERT_TRUE(knownAt(memory, "(in3[2,s,m,4])", "main"));
    ASSERT_FALSE(knownAt(memory, "(in3[2,r2,n,4])", "main"));
    ASSERT_TRUE((*memory.lbMemory.intScopeGroups.find(static_cast<int64_t>(group)) & gl::kIntScopeGroupPromoted) != 0);
    ASSERT_TRUE(memory.pendingWipeScopes.lookup(p) != 0);
    ASSERT_TRUE(memory.pendingWipeScopes.lookup(q) != 0);
    ASSERT_TRUE(memory.pendingWipeScopes.lookup(p2) != 0);
    ASSERT_EQ(memory.pendingWipeScopes.count(), 3);

    const int32_t levelsId = gl::lookupStatementLevels(memory.intStatementLevelsMap, memory.nameMap,
        gl::StrSpan(std::string("(in3[2,s,m,4])")), gl::StrSpan("main", 4));
    ASSERT_TRUE(levelsId != 0);
    ASSERT_EQ(memory.intStatementLevelsMap.runLen(levelsId), 1);
    ASSERT_EQ(memory.intStatementLevelsMap.valueAt(levelsId, 0), 5);
    int64_t mainKey = 0, leafKey = 0;
    ASSERT_TRUE(gl::lookupOriginKey(memory.originInterner, "(in3[2,r,m,4])", "main", mainKey));
    ASSERT_TRUE(gl::lookupOriginKey(memory.originInterner, "(in3[2,r,m,4])", scopeP, leafKey));
    bool cited = false;
    for (const gl::IdOrigin& origin : memory.exprOriginMap.recordsAt(memory.exprOriginMap.lookup(mainKey)))
        cited |= origin.first == gl::OriginTag::intScope && origin.second.size() == 1 && origin.second[0] == leafKey;
    ASSERT_TRUE(cited);

    const gl::NameId wiped[3] = { p, p2, q };
    memory.wipeSubtrees(wiped, 3);
    ASSERT_TRUE(memory.lbMemory.intScopes.find(static_cast<int64_t>(p)) == nullptr);
    ASSERT_TRUE(memory.lbMemory.intScopes.find(static_cast<int64_t>(q)) == nullptr);
    ASSERT_TRUE(memory.lbMemory.intScopes.find(static_cast<int64_t>(p2)) == nullptr);
    ASSERT_EQ(memory.lbMemory.intScopes.count(), 0);
    ASSERT_TRUE(memory.lbMemory.intScopeFirstFinish.find(gl::packInt32Pair(group, 2)) != nullptr);
    ASSERT_TRUE(knownAt(memory, "(in3[2,s,m,4])", "main"));
    ASSERT_FALSE(knownAt(memory, "(in3[2,s,m,4])", scopeQ));
    ASSERT_FALSE(knownAt(memory, "(in3[2,r2,n,4])", scopeP2));

    const int32_t namesBefore = memory.nameMap.nameCount();
    ASSERT_EQ(gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiQ, sp("s2")), gl::NameId(0));
    ASSERT_EQ(gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiP, sp("r3")), gl::NameId(0));
    ASSERT_EQ(memory.nameMap.nameCount(), namesBefore);
    ASSERT_EQ(memory.lbMemory.intScopes.count(), 0);
}

// A wipe that closes a premise's first finished scope before its group is
// promoted drops that first finish: the other premise's finish then promotes
// nothing, and the premise's next finished scope counts and promotes the
// group.
TEST(int_scope, wipe_drops_the_first_finish_of_an_open_group) {
    IntScopeFixture f;
    gl::Memory& memory = f.memory;
    auto& gate = f.analyzer.admissionMemory;
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    gl::WorkInstruction wiP(&arena);
    gl::WorkInstruction wiQ(&arena);
    boundInstruction(wiP, "(in3[2,marker1,marker2,4])", kCopyP);
    boundInstruction(wiQ, "(in3[2,marker2,marker1,4])", kCopyQ);
    const gl::NameId p = gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiP, sp("r"));
    const gl::NameId q = gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiQ, sp("s"));
    const gl::NameId q2 = gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiQ, sp("s2"));
    const std::string scopeP = memory.nameMap.decode(p);
    const std::string scopeQ = memory.nameMap.decode(q);
    const std::string scopeQ2 = memory.nameMap.decode(q2);
    const gl::NameId group = memory.nameMap.lookup(gl::StrSpan(std::string("(in3[2,marker,marker,4])")));
    ASSERT_TRUE(group != 0);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);

    addAt(f.analyzer, "(in3[2,s,m,4])", scopeQ, 5, out);
    const int64_t* firstQ = memory.lbMemory.intScopeFirstFinish.find(gl::packInt32Pair(group, 2));
    ASSERT_TRUE(firstQ != nullptr && *firstQ == static_cast<int64_t>(q));
    const gl::NameId wiped[1] = { q };
    memory.wipeSubtrees(wiped, 1);
    ASSERT_TRUE(memory.lbMemory.intScopeFirstFinish.find(gl::packInt32Pair(group, 2)) == nullptr);

    addAt(f.analyzer, "(in3[2,r,m,4])", scopeP, 4, out);
    const int64_t* firstP = memory.lbMemory.intScopeFirstFinish.find(gl::packInt32Pair(group, 1));
    ASSERT_TRUE(firstP != nullptr && *firstP == static_cast<int64_t>(p));
    ASSERT_FALSE(knownAt(memory, "(in3[2,r,m,4])", "main"));
    ASSERT_TRUE((*memory.lbMemory.intScopeGroups.find(static_cast<int64_t>(group)) & gl::kIntScopeGroupPromoted) == 0);

    addAt(f.analyzer, "(in3[2,s2,n,4])", scopeQ2, 5, out);
    ASSERT_TRUE(knownAt(memory, "(in3[2,r,m,4])", "main"));
    ASSERT_TRUE(knownAt(memory, "(in3[2,s2,n,4])", "main"));
    ASSERT_FALSE(knownAt(memory, "(in3[2,s,m,4])", "main"));
    ASSERT_TRUE((*memory.lbMemory.intScopeGroups.find(static_cast<int64_t>(group)) & gl::kIntScopeGroupPromoted) != 0);
}

// In a weak step block a key is connected through `rec`, the induction
// variable 9, a size marker, the predecessor of `rec` or the successor of 9
// known at the filing scope or above; the marker and the wildcard never
// connect, and neither does a successor link in the other direction.
TEST(weak_step_key, connected_arguments) {
    IntScopeFixture f;
    f.memory.weakStepBlock = true;
    auto& gate = f.analyzer.admissionMemory;
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(sequence[u_1,u_4,u_2,u_rec,marker])"), sp("main")));
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(fold[u_1,u_9,marker])"), sp("main")));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(ordered[u_1,u_4,marker])"), sp("main")));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(in3[u_2,marker,pi_,u_4])"), sp("main")));

    const gl::StrSpan fxy = sp("(fXY[marker,u_int_0_lev_3_1,u_1])");
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, fxy, sp("main")));
    gate.recordSizeMarker(f.memory, sp("int_0_lev_3_1"));
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, fxy, sp("main")));
    gate.recordSizeMarker(f.memory, sp("int_0_lev_3_1"));
    ASSERT_EQ(f.memory.lbMemory.sizeMarkers.count(), 1);

    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(p[u_x,marker])"), sp("main")));
    addAt(f.analyzer, "(in2[x,rec,3])", "main", 3, out);
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(p[u_x,marker])"), sp("main")));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(p[u_y,marker])"), sp("main")));
    addAt(f.analyzer, "(in2[9,y,3])", "main", 3, out);
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(p[u_y,marker])"), sp("main")));
    addAt(f.analyzer, "(in2[z,9,3])", "main", 3, out);
    addAt(f.analyzer, "(in2[rec,v,3])", "main", 3, out);
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(p[u_z,marker])"), sp("main")));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(p[u_v,marker])"), sp("main")));

    // A link known only at a child scope connects keys filed there only.
    const std::string child = f.memory.nameMap.decode(f.memory.nameMap.encodePush(gl::NameMap::MAIN_ID,
        std::string("(implication3[1,2])")));
    f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, f.memory.nameMap.lookup(gl::StrSpan(child)));
    addAt(f.analyzer, "(in2[x2,rec,3])", child, 3, out);
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(p[u_x2,marker])"), sp("main")));
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(p[u_x2,marker])"), gl::StrSpan(child)));
}

// A route-3 commit records its witness as a size marker; the continuation
// key that holds the witness is filed, the one that holds nothing connected
// is not. Outside a weak step block both keys name a head variable of the
// goal `(goal74[1,2,3])` and are filed, and the witness, admitted through a
// key naming the head variable 2, is recorded as a connected value
// (D-593).
TEST(weak_step_key, continuation_keys_need_a_connection) {
    for (const bool weak : { true, false }) {
        IntScopeFixture f;
        f.memory.weakStepBlock = weak;
        auto& gate = f.analyzer.admissionMemory;
        const std::string key = "(in2[u_2,marker,u_3])";
        fileInstruction(f.memory, key, "main", "(in3[u_2,marker,pi_x,u_4])");
        fileInstruction(f.memory, key, "main", "(in3[u_2,u_3,pi_x,u_5])");
        gl::InductionScopeSink opened;
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
            sp("main"), true, 0, nullptr, 0, nullptr, &opened));
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_w,marker,u_4])", "main"));
        ASSERT_EQ(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_5])", "main"), !weak);
        const int32_t witness = f.memory.valueInterner.lookup(sp("w"));
        ASSERT_TRUE(witness != 0);
        ASSERT_TRUE(f.memory.lbMemory.sizeMarkers.lookup(witness) != 0);
    }
}

// In a strong step block a key is connected exactly as in a weak one, and a
// value the strong demands caught connects once recorded as a size marker; a
// key holding nothing connected, a singleton's `ordered` key among them, is
// not connected (D-584).
TEST(strong_step_key, connected_arguments) {
    IntScopeFixture f;
    f.memory.strongStepBlock = true;
    auto& gate = f.analyzer.admissionMemory;
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(sequence[u_1,u_4,u_2,u_rec,marker])"), sp("main")));
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(fold[u_1,u_9,marker])"), sp("main")));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(ordered[u_1,u_4,marker])"), sp("main")));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(in3[u_2,marker,pi_,u_4])"), sp("main")));

    const gl::StrSpan caught = sp("(sequence[u_1,u_4,u_2,u_int_0_lev_3_9,marker])");
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, caught, sp("main")));
    gate.recordSizeMarker(f.memory, sp("int_0_lev_3_9"));
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, caught, sp("main")));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(fXY[u_int_0_lev_3_15,marker,u_1])"), sp("main")));
}

// A size marker connects only where the filing scope does not know it equal
// to the induction's zero (`2` in the fixture): there a key over it holds a
// size of zero, which bounds nothing. The test reads the scope's classes, so
// the zero itself recorded as a size marker does not connect either; both
// step kinds (D-584).
TEST(strong_step_key, size_marker_equal_to_zero_does_not_connect) {
    for (const bool strong : { true, false }) {
        IntScopeFixture f;
        f.memory.strongStepBlock = strong;
        f.memory.weakStepBlock = !strong;
        auto& gate = f.analyzer.admissionMemory;
        gate.recordSizeMarker(f.memory, sp("m"));
        const std::string reductio = f.memory.nameMap.decode(f.memory.nameMap.encodePush(gl::NameMap::MAIN_ID,
            std::string("(implication3[1,2])")));
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, f.memory.nameMap.lookup(gl::StrSpan(reductio)));
        const gl::StrSpan key = sp("(sequence[u_1,u_4,u_2,u_m,marker])");
        ASSERT_TRUE(gate.stepKeyConnected(f.memory, key, sp("main")));
        ASSERT_TRUE(gate.stepKeyConnected(f.memory, key, gl::StrSpan(reductio)));

        gl::DirtyState dirty = gl::DirtyState::Clean;
        gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
        addAt(f.analyzer, "(=[m,2])", reductio, 3, out);
        ASSERT_FALSE(gate.stepKeyConnected(f.memory, key, gl::StrSpan(reductio)));
        ASSERT_TRUE(gate.stepKeyConnected(f.memory, key, sp("main")));

        gate.recordSizeMarker(f.memory, sp("2"));
        ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(interval[u_1,u_4,u_2,u_2,marker])"), sp("main")));
    }
}

// The step-key rule binds every scope of a weak step block, and in a strong
// step block an induction scope and the scopes below it; `main` and a
// subproof beside the induction scope file freely, as in every other block
// (D-584).
TEST(strong_step_key, rule_binds_at_induction_scopes) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    gl::NameMap& nm = f.memory.nameMap;
    const gl::NameId inductionId = nm.encodePush(gl::NameMap::MAIN_ID,
        std::string("is_(in3[marker1,marker2,u_9,u_4])!(=[marker1,u_9])_m_x"));
    const std::string induction = nm.decode(inductionId);
    const std::string below = nm.decode(nm.encodePush(inductionId, std::string("(implication3[1,2])")));
    const std::string beside = nm.decode(nm.encodePush(gl::NameMap::MAIN_ID, std::string("(implication4[1,2])")));
    const std::string scopes[4] = { "main", beside, induction, below };
    for (const std::string& scope : scopes)
        ASSERT_FALSE(gate.stepKeyRuleBinds(f.memory, gl::StrSpan(scope)));
    f.memory.weakStepBlock = true;
    for (const std::string& scope : scopes)
        ASSERT_TRUE(gate.stepKeyRuleBinds(f.memory, gl::StrSpan(scope)));
    f.memory.weakStepBlock = false;
    f.memory.strongStepBlock = true;
    ASSERT_FALSE(gate.stepKeyRuleBinds(f.memory, sp("main")));
    ASSERT_FALSE(gate.stepKeyRuleBinds(f.memory, gl::StrSpan(beside)));
    ASSERT_TRUE(gate.stepKeyRuleBinds(f.memory, gl::StrSpan(induction)));
    ASSERT_TRUE(gate.stepKeyRuleBinds(f.memory, gl::StrSpan(below)));
}

// Every key of a weak step block admits size markers; in a strong step block
// only a key connected at its own scope does, through the induction variable
// or a size marker (D-584).
TEST(strong_step_key, admitting_key_connected) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const int64_t ordered = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        "(ordered[u_1,u_4,marker])", "main");
    const int64_t fold = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        "(fold[u_1,u_9,marker])", "main");
    const int64_t caught = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        "(sequence[u_1,u_4,u_2,u_int_0_lev_3_9,marker])", "main");
    f.memory.weakStepBlock = true;
    ASSERT_TRUE(gate.admittingKeyConnected(f.memory, ordered));
    f.memory.weakStepBlock = false;
    f.memory.strongStepBlock = true;
    ASSERT_FALSE(gate.admittingKeyConnected(f.memory, ordered));
    ASSERT_TRUE(gate.admittingKeyConnected(f.memory, fold));
    ASSERT_FALSE(gate.admittingKeyConnected(f.memory, caught));
    gate.recordSizeMarker(f.memory, sp("int_0_lev_3_9"));
    ASSERT_TRUE(gate.admittingKeyConnected(f.memory, caught));
}

// At `main` of a strong step block a route-3 commit through a connected key
// records its witness, and both continuation keys are filed, the one holding
// nothing connected too: `main` files freely. At an induction scope a commit
// through a connected key records its witness; the continuation key holding
// the witness is filed, the one holding nothing connected is not (D-584).
TEST(strong_step_key, continuation_keys_need_a_connection) {
    {
        IntScopeFixture f;
        f.memory.strongStepBlock = true;
        f.memory.premiseSplitFiled = false;
        auto& gate = f.analyzer.admissionMemory;
        const std::string key = "(in2[u_9,marker,u_3])";
        fileInstruction(f.memory, key, "main", "(in3[u_2,marker,pi_x,u_4])", key);
        fileInstruction(f.memory, key, "main", "(in3[u_2,u_3,pi_x,u_5])", key);
        gl::InductionScopeSink opened;
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[9,w,3])"), sp("w"), sp("(in2[9,marker,3])"),
            sp("main"), true, 0, nullptr, 0, nullptr, &opened));
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_w,marker,u_4])", "main"));
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_5])", "main"));
        const int32_t witness = f.memory.valueInterner.lookup(sp("w"));
        ASSERT_TRUE(witness != 0 && f.memory.lbMemory.sizeMarkers.lookup(witness) != 0);
    }
    {
        IntScopeFixture f;
        f.memory.strongStepBlock = true;
        f.memory.premiseSplitFiled = false;
        auto& gate = f.analyzer.admissionMemory;
        const gl::NameId inductionId = f.memory.nameMap.encodePush(gl::NameMap::MAIN_ID,
            std::string("is_(in3[marker1,marker2,u_9,u_4])!(=[marker1,u_9])_m_x"));
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, inductionId);
        const std::string scope = f.memory.nameMap.decode(inductionId);
        const std::string key = "(in2[u_9,marker,u_3])";
        fileInstruction(f.memory, key, scope, "(in3[u_2,marker,pi_x,u_4])", key);
        fileInstruction(f.memory, key, scope, "(in3[u_2,u_3,pi_x,u_5])", key);
        gl::InductionScopeSink opened;
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[9,w,3])"), sp("w"), sp("(in2[9,marker,3])"),
            gl::StrSpan(scope), true, 0, nullptr, 0, nullptr, &opened));
        const int32_t witness = f.memory.valueInterner.lookup(sp("w"));
        ASSERT_TRUE(witness != 0 && f.memory.lbMemory.sizeMarkers.lookup(witness) != 0);
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_w,marker,u_4])", scope));
        ASSERT_FALSE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_5])", scope));
    }
}

// Outside the induction scopes of a strong step block a key that holds
// nothing connected admits nothing: at `main` the route-3 probe, the route-3
// commit and the statement fill refuse it, while a connected key admits
// there. The same key filed at `main` still admits at an induction scope it
// serves (D-572), and in every other block it admits everywhere
// (D-602).
TEST(strong_step_key, main_admits_only_through_connected_keys) {
    const std::string unconnected = "(in2[u_2,marker,u_3])";
    const std::string connected = "(in2[u_9,marker,u_3])";
    {
        IntScopeFixture f;
        f.memory.strongStepBlock = true;
        f.memory.premiseSplitFiled = false;
        auto& gate = f.analyzer.admissionMemory;
        fileInstruction(f.memory, unconnected, "main", "(in3[u_2,marker,pi_x,u_4])");
        fileInstruction(f.memory, connected, "main", "(in3[u_2,marker,pi_x,u_4])", connected);
        const int64_t unconnectedPk = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
            unconnected, "main");
        const int64_t connectedPk = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
            connected, "main");
        ASSERT_FALSE(gate.strongStepKeyAdmits(f.memory, unconnectedPk, gl::NameMap::MAIN_ID));
        ASSERT_TRUE(gate.strongStepKeyAdmits(f.memory, connectedPk, gl::NameMap::MAIN_ID));

        gl::InductionScopeSink opened;
        ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
            sp("main"), false));
        ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
            sp("main"), true, 0, nullptr, 0, nullptr, &opened));
        ASSERT_TRUE(copyValues(f.memory, unconnected, "main", 0).empty());
        ASSERT_FALSE(keyFiled(f.memory, "(in3[u_2,u_w,marker,u_4])", "main"));

        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[9,v,3])"), sp("v"), sp("(in2[9,marker,3])"),
            sp("main"), false));
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[9,v,3])"), sp("v"), sp("(in2[9,marker,3])"),
            sp("main"), true, 0, nullptr, 0, nullptr, &opened));
        ASSERT_TRUE(copyValues(f.memory, connected, "main", 0) == std::vector<std::string>{ "v" });
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_v,marker,u_4])", "main"));

        gate.updateAdmissionMapIntegration(sp("(in2[2,t,3])"), f.memory, sp("main"), 0, 0, true);
        ASSERT_TRUE(copyValues(f.memory, unconnected, "main", 0).empty());
        ASSERT_FALSE(keyFiled(f.memory, "(in3[u_2,u_t,marker,u_4])", "main"));
        gate.updateAdmissionMapIntegration(sp("(in2[9,u,3])"), f.memory, sp("main"), 0, 0, true);
        ASSERT_EQ(copyValues(f.memory, connected, "main", 0).size(), static_cast<std::size_t>(2));
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_u,marker,u_4])", "main"));
    }
    {
        IntScopeFixture f;
        f.memory.strongStepBlock = true;
        f.memory.premiseSplitFiled = false;
        auto& gate = f.analyzer.admissionMemory;
        const gl::NameId inductionId = f.memory.nameMap.encodePush(gl::NameMap::MAIN_ID,
            std::string("is_(in3[marker1,marker2,u_9,u_4])!(=[marker1,u_9])_m_x"));
        f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, inductionId);
        const std::string scope = f.memory.nameMap.decode(inductionId);
        fileInstruction(f.memory, unconnected, "main", "(in3[u_2,marker,pi_x,u_4])");
        const int64_t pk = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap, unconnected, "main");
        ASSERT_TRUE(gate.strongStepKeyAdmits(f.memory, pk, inductionId));
        gl::InductionScopeSink opened;
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
            gl::StrSpan(scope), false));
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
            gl::StrSpan(scope), true, 0, nullptr, 0, nullptr, &opened));
    }
    for (const bool weak : { true, false }) {
        IntScopeFixture f;
        f.memory.weakStepBlock = weak;
        f.memory.premiseSplitFiled = false;
        auto& gate = f.analyzer.admissionMemory;
        fileInstruction(f.memory, unconnected, "main", "(in3[u_2,marker,pi_x,u_4])");
        const int64_t pk = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap, unconnected, "main");
        ASSERT_TRUE(gate.strongStepKeyAdmits(f.memory, pk, gl::NameMap::MAIN_ID));
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
            sp("main"), false));
    }
}

// The statement fill records the value it fills as a size marker, and the
// continuation key holding it is filed.
TEST(weak_step_key, statement_fill_records_a_size_marker) {
    IntScopeFixture f;
    f.memory.weakStepBlock = true;
    auto& gate = f.analyzer.admissionMemory;
    fileInstruction(f.memory, "(in2[u_2,marker,u_3])", "main", "(in3[u_2,marker,pi_x,u_4])");
    gate.updateAdmissionMapIntegration(sp("(in2[2,t,3])"), f.memory, sp("main"), 0, 0, true);
    const int32_t value = f.memory.valueInterner.lookup(sp("t"));
    ASSERT_TRUE(value != 0 && f.memory.lbMemory.sizeMarkers.lookup(value) != 0);
    ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_t,marker,u_4])", "main"));
}

// Outside the step blocks a key is connected to its goal through a head
// variable (an argument of the goal that is neither an anchor argument, of
// `(AnchorTest92[N,0,s])` in the fixture, nor a label) or through a connected
// value. Anchor arguments, the marker labels and `pi_` names take no part, an
// anchor argument recorded as a value included; a goal without head
// variables files every key (D-593).
TEST(goal_key, connected_arguments) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const gl::StrSpan goal = sp("(goal77[N,0,s,9])");
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, sp("(ordered[u_N,u_0,marker])"), goal, gl::NameMap::MAIN_ID));
    ASSERT_TRUE(gate.goalKeyConnected(f.memory, sp("(fold[u_N,u_9,marker])"), goal, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, sp("(in3[u_s,marker,pi_,u_0])"), goal, gl::NameMap::MAIN_ID));

    const gl::StrSpan fxy = sp("(fXY[marker,u_w,u_N])");
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, fxy, goal, gl::NameMap::MAIN_ID));
    gate.recordSizeMarker(f.memory, sp("w"));
    ASSERT_TRUE(gate.goalKeyConnected(f.memory, fxy, goal, gl::NameMap::MAIN_ID));
    gate.recordSizeMarker(f.memory, sp("0"));
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, sp("(ordered[u_N,u_0,marker])"), goal, gl::NameMap::MAIN_ID));

    ASSERT_TRUE(gate.goalKeyConnected(f.memory, sp("(ordered[u_N,u_0,marker])"), sp("(goal78[N,0,s])"), gl::NameMap::MAIN_ID));
    ASSERT_TRUE(gate.goalKeyConnected(f.memory, sp("(ordered[u_N,u_0,marker])"), sp("(goal79[marker1,0,marker2])"), gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, sp("(ordered[u_N,u_0,marker])"), sp("(goal79[marker1,9,marker2])"), gl::NameMap::MAIN_ID));
    ASSERT_TRUE(gate.goalKeyConnected(f.memory, sp("(p[u_9,marker])"), sp("(goal79[marker1,9,marker2])"), gl::NameMap::MAIN_ID));
}

// A key that names only anchor arguments connects when it holds an anchor
// argument exactly as often as the goal does, more than once: a goal over
// [0,0] keys the interval [0,0]. The same key holding it once or three times,
// or for a goal that holds it once, does not connect
// (D-593).
TEST(goal_key, repeated_anchor_argument_connects) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const gl::StrSpan goal = sp("(sequenceTest[N,0,0,X])");
    ASSERT_TRUE(gate.goalKeyConnected(f.memory, sp("(interval[u_N,u_0,u_0,marker])"), goal, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, sp("(interval[u_N,u_0,u_s,marker])"), goal, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, sp("(triple[u_0,u_0,u_0,marker])"), goal, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, sp("(ordered[u_N,u_s,marker])"), goal, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.goalKeyConnected(f.memory, sp("(interval[u_N,u_0,u_0,marker])"), sp("(goalTest[N,0,X])"),
        gl::NameMap::MAIN_ID));
}

// The admitting key's own text decides whether the value it admits becomes a
// connected value: a key naming a head variable of the accepting
// instruction's goal does, one naming only anchor arguments does not unless
// the goal has no head variable (D-593).
TEST(goal_key, admitting_key_connected_to_goal) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const int64_t ordered = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        "(ordered[u_N,u_0,marker])", "main");
    const int64_t fold = gl::mintTemplateKey(f.memory.templateInterner, f.memory.nameMap,
        "(fold[u_N,u_9,marker])", "main");
    ASSERT_FALSE(gate.admittingKeyConnectedToGoal(f.memory, ordered, sp("(goal77[N,0,s,9])")));
    ASSERT_TRUE(gate.admittingKeyConnectedToGoal(f.memory, fold, sp("(goal77[N,0,s,9])")));
    ASSERT_TRUE(gate.admittingKeyConnectedToGoal(f.memory, ordered, sp("(goal78[N,0,s])")));
}

// Outside the step blocks a route-3 commit through a key naming the goal's
// head variable 9 records its witness as a connected value: the continuation
// key that holds the witness is filed and so is the one naming 9, while the
// one naming only anchor arguments is not. A commit through a key naming no
// head variable still admits but records nothing, so its continuation key
// over the witness is not filed (D-593).
TEST(goal_key, continuation_keys_need_a_connection) {
    const std::string goal = "(goal77[N,0,s,9])";
    {
        IntScopeFixture f;
        f.memory.premiseSplitFiled = false;
        auto& gate = f.analyzer.admissionMemory;
        const std::string key = "(in2[u_9,marker,u_s])";
        fileInstruction(f.memory, key, "main", "(in3[u_N,marker,pi_x,u_0])", key, goal);
        fileInstruction(f.memory, key, "main", "(in3[u_N,u_0,pi_x,u_s])", key, goal);
        fileInstruction(f.memory, key, "main", "(in3[u_9,u_0,pi_x,u_s])", key, goal);
        gl::InductionScopeSink opened;
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[9,w,s])"), sp("w"), sp("(in2[9,marker,s])"),
            sp("main"), true, 0, nullptr, 0, nullptr, &opened));
        const int32_t witness = f.memory.valueInterner.lookup(sp("w"));
        ASSERT_TRUE(witness != 0 && f.memory.lbMemory.sizeMarkers.lookup(witness) != 0);
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_N,u_w,marker,u_0])", "main"));
        ASSERT_FALSE(keyFiled(f.memory, "(in3[u_N,u_0,marker,u_s])", "main"));
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_9,u_0,marker,u_s])", "main"));
    }
    {
        IntScopeFixture f;
        f.memory.premiseSplitFiled = false;
        auto& gate = f.analyzer.admissionMemory;
        const std::string key = "(in2[u_N,marker,u_s])";
        fileInstruction(f.memory, key, "main", "(in3[u_N,marker,pi_x,u_0])", key, goal);
        gl::InductionScopeSink opened;
        ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[N,w,s])"), sp("w"), sp("(in2[N,marker,s])"),
            sp("main"), true, 0, nullptr, 0, nullptr, &opened));
        ASSERT_TRUE(copyValues(f.memory, key, "main", 0) == std::vector<std::string>{ "w" });
        const int32_t witness = f.memory.valueInterner.lookup(sp("w"));
        ASSERT_TRUE(witness != 0 && f.memory.lbMemory.sizeMarkers.lookup(witness) == 0);
        ASSERT_FALSE(keyFiled(f.memory, "(in3[u_N,u_w,marker,u_0])", "main"));
    }
}

// The statement fill records the value it fills through a key naming a head
// variable of the goal as a connected value, and the continuation key holding
// it is filed; through a key naming none it fills, records nothing and files
// no continuation over the value (D-593).
TEST(goal_key, statement_fill_records_a_connected_value) {
    for (const bool connected : { true, false }) {
        IntScopeFixture f;
        f.memory.premiseSplitFiled = false;
        auto& gate = f.analyzer.admissionMemory;
        const std::string key = connected ? "(in2[u_9,marker,u_s])" : "(in2[u_N,marker,u_s])";
        fileInstruction(f.memory, key, "main", "(in3[u_N,marker,pi_x,u_0])", key, "(goal77[N,0,s,9])");
        gate.updateAdmissionMapIntegration(sp(connected ? "(in2[9,t,s])" : "(in2[N,t,s])"), f.memory,
            sp("main"), 0, 0, true);
        ASSERT_TRUE(copyValues(f.memory, key, "main", 0) == std::vector<std::string>{ "t" });
        const int32_t value = f.memory.valueInterner.lookup(sp("t"));
        ASSERT_TRUE(value != 0);
        ASSERT_EQ(f.memory.lbMemory.sizeMarkers.lookup(value) != 0, connected);
        ASSERT_EQ(keyFiled(f.memory, "(in3[u_N,u_t,marker,u_0])", "main"), connected);
    }
}

// A route-3 commit that writes its witness into an instruction records it as
// a one-way witness; a pure probe records nothing
// (D-585).
TEST(one_way, commit_records_its_witness) {
    IntScopeFixture f;
    auto& gate = f.analyzer.admissionMemory;
    fileInstruction(f.memory, "(in2[u_2,marker,u_3])", "main", "(in3[u_2,marker,pi_x,u_4])");
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
        sp("main"), false, 0, nullptr, 0, nullptr, nullptr));
    ASSERT_TRUE(f.memory.lbMemory.oneWayWitnesses.empty());
    gl::InductionScopeSink opened;
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,w,3])"), sp("w"), sp("(in2[2,marker,3])"),
        sp("main"), true, 0, nullptr, 0, nullptr, &opened));
    const gl::NameId witness = f.memory.nameMap.lookup(sp("w"));
    ASSERT_TRUE(witness != 0);
    ASSERT_EQ(f.memory.lbMemory.oneWayWitnesses.count(), 1);
    ASSERT_TRUE(f.memory.lbMemory.oneWayWitnesses.lookup(witness) != 0);
}

// The statement fill tries no recorded witness as the marker: its key gains no
// value and files no continuation, while another argument of the same
// statement still fills; without the record the witness fills as before
// (D-585).
TEST(one_way, statement_fill_skips_a_recorded_witness) {
    for (const bool recorded : { false, true }) {
        IntScopeFixture f;
        auto& gate = f.analyzer.admissionMemory;
        fileInstruction(f.memory, "(in2[u_2,marker,u_3])", "main", "(in3[u_2,marker,pi_x,u_4])");
        fileInstruction(f.memory, "(in2[marker,u_t,u_3])", "main", "(in3[u_2,u_3,pi_y,u_5])",
            "(in2[marker,u_t,u_3])");
        if (recorded) gate.recordOneWayWitness(f.memory, sp("t"));
        gate.updateAdmissionMapIntegration(sp("(in2[2,t,3])"), f.memory, sp("main"), 0, 0, true);
        ASSERT_EQ(keyFiled(f.memory, "(in3[u_2,u_t,marker,u_4])", "main"), !recorded);
        ASSERT_EQ(copyValues(f.memory, "(in2[u_2,marker,u_3])", "main", 0).empty(), recorded);
        ASSERT_TRUE(copyValues(f.memory, "(in2[marker,u_t,u_3])", "main", 0) == std::vector<std::string>{ "2" });
        ASSERT_TRUE(keyFiled(f.memory, "(in3[u_2,u_3,marker,u_5])", "main"));
    }
}

namespace {
    /// @brief The full I5's copy bound to P (level 1): forbids Q's own 12 and
    ///        13 and level 2.
    const gl::IntegrationForbiddenSets kTwinP = binding((uint64_t{1} << 12) | (uint64_t{1} << 13), uint64_t{1} << 2, 1);
    /// @brief The full I5's copy bound to Q (level 2): forbids P's own 9 and
    ///        10 and level 1.
    const gl::IntegrationForbiddenSets kTwinQ = binding((uint64_t{1} << 9) | (uint64_t{1} << 10), uint64_t{1} << 1, 2);

    /// @brief Turn the fixture into the full I5's weak step block: Q's block
    ///        is the second twin premise `(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])`,
    ///        12 the twin partner, 3 the successor, and `rec2` the recorded
    ///        name of 12's predecessor.
    void armTwinSecondGoal(IntScopeFixture& f) {
        f.premiseQ.setExprKey("(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])");
        f.memory.weakStepBlock = true;
        f.memory.twinPartnerId = f.memory.nameMap.encode(sp("12"));
        f.memory.twinSuccessorId = f.memory.nameMap.encode(sp("3"));
        f.memory.twinFirstGoalId = f.memory.nameMap.encode(sp("(in3[rec,marker1,marker2,4])"));
        f.memory.twinFirstLevel = 1;
        f.memory.twinSecondLevel = 2;
        f.memory.twinPredecessorId = f.memory.nameMap.encode(sp("rec2"));
    }
} // namespace

// With q in `rec`'s role, q is a number and `rec` is not.
TEST(twin_second_goal, numbers_with_a_rec_role) {
    ASSERT_TRUE(gl::isNumberArgument(sp("u_q"), sp("q")));
    ASSERT_TRUE(gl::isNumberArgument(sp("12"), sp("q")));
    ASSERT_FALSE(gl::isNumberArgument(sp("rec"), sp("q")));
    ASSERT_FALSE(gl::isNumberArgument(sp("q2"), sp("q")));
    ASSERT_TRUE(gl::isNumberArgument(sp("u_rec"), sp("rec")));
    ASSERT_TRUE(gl::hasOnlyNumberFreeArguments(sp("(existence12[1,4,q,12,2])"), sp("q")));
    ASSERT_FALSE(gl::hasOnlyNumberFreeArguments(sp("(existence12[1,4,rec,12,2])"), sp("q")));
    ASSERT_TRUE(gl::hasOnlyNumberFreeArguments(sp("!(>[I](interval[1,4,2,q,I])!(fXY[10,I,1]))"), sp("q")));
    ASSERT_FALSE(gl::hasOnlyNumberFreeArguments(sp("(p[u_1,int_0_lev_3_0])"), sp("q")));
}

// A key is connected through either named role, never through another name.
TEST(twin_second_goal, key_holds_induction_role) {
    ASSERT_TRUE(gl::ScopeSyntax::keyHoldsInductionRole(sp("(sequence[u_1,u_4,u_2,u_q,marker])"), sp("q"), sp("12")));
    ASSERT_TRUE(gl::ScopeSyntax::keyHoldsInductionRole(sp("(fold[u_1,u_12,marker])"), sp("q"), sp("12")));
    ASSERT_FALSE(gl::ScopeSyntax::keyHoldsInductionRole(sp("(sequence[u_1,u_4,u_2,u_rec,marker])"), sp("q"), sp("12")));
    ASSERT_FALSE(gl::ScopeSyntax::keyHoldsInductionRole(sp("(fold[u_1,u_qq,marker])"), sp("q"), sp("12")));
}

// The second twin premise's copy plays `rec2` and 12; every other
// instruction plays `rec` and the inducted 9.
TEST(twin_second_goal, roles_by_premise_level) {
    IntScopeFixture f;
    armTwinSecondGoal(f);
    const gl::InductionRoles unbound = gl::inductionRolesOf(f.memory, 0);
    ASSERT_TRUE(gl::equalSpans(unbound.rec, sp("rec")) && gl::equalSpans(unbound.variable, sp("9")));
    const gl::InductionRoles first = gl::inductionRolesOf(f.memory, 1);
    ASSERT_TRUE(gl::equalSpans(first.rec, sp("rec")) && gl::equalSpans(first.variable, sp("9")));
    const gl::InductionRoles second = gl::inductionRolesOf(f.memory, 2);
    ASSERT_TRUE(gl::equalSpans(second.rec, sp("rec2")) && gl::equalSpans(second.variable, sp("12")));
}

// Each twin copy forbids the other premise's own variables and level.
TEST(twin_second_goal, twin_premise_forbidden_sets) {
    IntScopeFixture f;
    armTwinSecondGoal(f);
    gl::IntegrationForbiddenSets p;
    gl::IntegrationForbiddenSets q;
    gl::twinPremiseForbiddenSets(f.memory, 1, 2, 0, p);
    gl::twinPremiseForbiddenSets(f.memory, 1, 2, 1, q);
    ASSERT_EQ(gl::compareForbiddenSets(p, kTwinP), 0);
    ASSERT_EQ(gl::compareForbiddenSets(q, kTwinQ), 0);
}

// The second copy starts only from numbers or `rec2`, and files its keys
// through `rec2`, 12, a predecessor of `rec2` or a successor of 12; the first
// copy keeps `rec` and 9.
TEST(twin_second_goal, second_copy_gates) {
    IntScopeFixture f;
    armTwinSecondGoal(f);
    auto& gate = f.analyzer.admissionMemory;
    ASSERT_FALSE(gl::premiseCopyRefuses(f.memory, gl::NameMap::MAIN_ID, kTwinQ, sp("(existence12[1,4,rec2,12,2])")));
    ASSERT_TRUE(gl::premiseCopyRefuses(f.memory, gl::NameMap::MAIN_ID, kTwinQ, sp("(existence12[1,4,rec,12,2])")));
    ASSERT_TRUE(gl::premiseCopyRefuses(f.memory, gl::NameMap::MAIN_ID, kTwinQ, sp("(existence12[1,4,rec2,10,2])")));
    ASSERT_FALSE(gl::premiseCopyRefuses(f.memory, gl::NameMap::MAIN_ID, kTwinP, sp("(existence12[1,4,rec,10,2])")));
    ASSERT_TRUE(gl::premiseCopyRefuses(f.memory, gl::NameMap::MAIN_ID, kTwinP, sp("(existence12[1,4,rec2,10,2])")));
    ASSERT_TRUE(gl::premiseCopyRefuses(f.memory, gl::NameMap::MAIN_ID, kTwinP, sp("(existence12[1,4,rec,13,2])")));

    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(sequence[u_1,u_4,u_2,u_rec2,marker])"), sp("main"), 2));
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(fold[u_1,u_12,marker])"), sp("main"), 2));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(sequence[u_1,u_4,u_2,u_rec,marker])"), sp("main"), 2));
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(sequence[u_1,u_4,u_2,u_rec,marker])"), sp("main"), 1));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(sequence[u_1,u_4,u_2,u_rec2,marker])"), sp("main"), 1));

    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    addAt(f.analyzer, "(in2[x,rec2,3])", "main", 3, out);
    addAt(f.analyzer, "(in2[12,y,3])", "main", 3, out);
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(p[u_x,marker])"), sp("main"), 2));
    ASSERT_TRUE(gate.stepKeyConnected(f.memory, sp("(p[u_y,marker])"), sp("main"), 2));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(p[u_x,marker])"), sp("main"), 1));
    ASSERT_FALSE(gate.stepKeyConnected(f.memory, sp("(p[u_y,marker])"), sp("main"), 1));
}

// The second copy's int scope joins the first goal's group (`rec2` read as
// `rec`) and finishes on the goal with `rec2`; the first finished scope of
// each premise is promoted together.
TEST(twin_second_goal, int_scope_group_and_finish) {
    IntScopeFixture f;
    armTwinSecondGoal(f);
    gl::Memory& memory = f.memory;
    auto& gate = f.analyzer.admissionMemory;
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    gl::WorkInstruction wiP(&arena);
    gl::WorkInstruction wiQ(&arena);
    boundInstruction(wiP, "(in3[rec,marker1,marker2,4])", kTwinP);
    boundInstruction(wiQ, "(in3[rec2,marker1,marker2,4])", kTwinQ);
    const gl::NameId p = gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiP, sp("r"));
    const gl::NameId q = gate.openIntScope(memory, gl::NameMap::MAIN_ID, wiQ, sp("s"));
    ASSERT_TRUE(p != 0 && q != 0);
    ASSERT_EQ(memory.nameMap.decode(q),
        std::string("main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])_s"));
    const gl::NameId group = memory.nameMap.lookup(gl::StrSpan(std::string("(in3[rec,marker,marker,4])")));
    ASSERT_TRUE(group != 0);
    ASSERT_EQ(memory.nameMap.lookup(gl::StrSpan(std::string("(in3[rec2,marker,marker,4])"))), gl::NameId(0));
    ASSERT_EQ(*memory.lbMemory.intScopes.find(static_cast<int64_t>(q)), gl::packInt32Pair(group, 2));
    ASSERT_EQ(*memory.lbMemory.intScopes.find(static_cast<int64_t>(p)), gl::packInt32Pair(group, 1));

    const std::string scopeP = memory.nameMap.decode(p);
    const std::string scopeQ = memory.nameMap.decode(q);
    gl::DirtyState dirty = gl::DirtyState::Clean;
    gl::PagedVector<gl::IntEncodedExpr> out(&gl::genScratchArenas().forSlot(0), &dirty);
    addAt(f.analyzer, "(in3[rec,a,b,4])", scopeQ, 3, out);
    ASSERT_TRUE(memory.lbMemory.intScopeFirstFinish.find(gl::packInt32Pair(group, 2)) == nullptr);
    addAt(f.analyzer, "(in3[rec2,a,b,4])", scopeQ, 3, out);
    const int64_t* firstQ = memory.lbMemory.intScopeFirstFinish.find(gl::packInt32Pair(group, 2));
    ASSERT_TRUE(firstQ != nullptr && *firstQ == static_cast<int64_t>(q));
    ASSERT_FALSE(knownAt(memory, "(in3[rec2,a,b,4])", "main"));
    addAt(f.analyzer, "(in3[rec,c,d,4])", scopeP, 3, out);
    ASSERT_TRUE(knownAt(memory, "(in3[rec2,a,b,4])", "main"));
    ASSERT_TRUE(knownAt(memory, "(in3[rec,c,d,4])", "main"));
    ASSERT_TRUE((*memory.lbMemory.intScopeGroups.find(static_cast<int64_t>(group)) & gl::kIntScopeGroupPromoted) != 0);
}

// In the twin successor case the statement fill reaches the copies bound to
// the second twin premise only: an arrival that is not numbers or `rec2` is
// refused, an arrival of numbers or `rec2` opens the copy's int scope below
// the case and continues there under a key connected through `rec2`, and the
// first premise's copy filed in the case stays untouched. Another case scope
// fills nothing.
TEST(twin_second_goal, statement_fill_in_the_twin_case) {
    IntScopeFixture f;
    armTwinSecondGoal(f);
    auto& gate = f.analyzer.admissionMemory;
    const std::string caseScope = "main_boundary_ordis_(or1[1,12,3,2])_((inSequence[1,12,3]))";
    const std::string zeroCase = "main_boundary_ordis_(or1[1,12,3,2])_((=[12,2]))";
    f.memory.twinCaseScopeId = f.memory.nameMap.encode(gl::StrSpan(caseScope));
    f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, f.memory.twinCaseScopeId);
    f.analyzer.canonicalizer.initializeCanonicalScope(f.memory, f.memory.nameMap.encode(gl::StrSpan(zeroCase)));
    const std::string key = "(in2[u_2,marker,u_3])";
    const std::string second = "(in3[u_2,marker,pi_x,u_4])";
    const std::string next = "(in3[u_2,u_rec2,marker,u_4])";
    fileTwoPartCopies(f.memory, key, caseScope, { kTwinP, kTwinQ }, second);
    fileTwoPartCopies(f.memory, key, zeroCase, { kTwinQ }, second);

    gate.updateAdmissionMapIntegration(sp("(in2[2,int_0_lev_3_5,3])"), f.memory, gl::StrSpan(caseScope),
        0, 0, false, false);
    ASSERT_EQ(f.memory.lbMemory.intScopes.count(), 0);

    gate.updateAdmissionMapIntegration(sp("(in2[2,rec2,3])"), f.memory, gl::StrSpan(caseScope), 0, 0, false, true);
    const std::string scopeQ = caseScope + "_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,12,13,11])_rec2";
    const std::string scopeP = caseScope + "_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])_rec2";
    ASSERT_TRUE(keyFiled(f.memory, next, scopeQ));
    ASSERT_FALSE(keyFiled(f.memory, next, scopeP));
    ASSERT_FALSE(keyFiled(f.memory, next, caseScope));
    ASSERT_EQ(f.memory.lbMemory.intScopes.count(), 1);
    const gl::NameId opened = f.memory.nameMap.lookup(gl::StrSpan(scopeQ));
    ASSERT_TRUE(opened != 0);
    ASSERT_EQ(f.memory.nameMap.parentOf(opened), f.memory.twinCaseScopeId);
    ASSERT_TRUE(copyValues(f.memory, key, caseScope, 1).empty());

    gate.updateAdmissionMapIntegration(sp("(in2[2,rec2,3])"), f.memory, gl::StrSpan(zeroCase), 0, 0, false, true);
    ASSERT_TRUE(copyValues(f.memory, key, zeroCase, 2).empty());
    ASSERT_EQ(f.memory.lbMemory.intScopes.count(), 1);
}
