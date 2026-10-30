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
/// @brief Unit tests for the witness-link admission gate
///        (D-529).
///
/// @details
/// In an induction step block every stored integration instruction carries
/// its path list: the non-anchor fixed arguments of its integration goal and
/// every value that filled a marker on the way, sorted, duplicate-free and
/// canonical. A route refuses an instruction whose list holds an argument
/// that the other part of an admitted witness's existence links the witness
/// to (same definition set as the witness, anchors excluded), compared
/// through the equivalence classes. Covers the list text, its storage in the
/// instruction records, the anchor and typing helpers, the gate predicate,
/// the route probe and the refusal marks that close a parked source's
/// demand. The disintegration-level cases live beside the integration trial
/// tests (`test_integration_trial.cpp`), whose fixture they share.

#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "canonical_test_helpers.hpp"
#include "../logical_block/scope_syntax.hpp"
#include "../prover.hpp"

#include <algorithm>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {
    /// @brief Span over a string literal (tests only).
    gl::StrSpan sp(const char* s) {
        return gl::StrSpan(s, static_cast<int32_t>(std::strlen(s)));
    }

    /// @brief Owned copy of a span (tests only).
    std::string str(gl::StrSpan s) {
        return std::string(s.ptr, static_cast<std::size_t>(s.len));
    }

    /// @brief The tests' string-tier scratch arena.
    gl::ScratchArena& strings() {
        return gl::scratchArenas().forSlot(gl::scratchArenas().slotCount() - 1);
    }

    /// @brief The tests' page-tier scratch arena.
    gl::ScratchArena& pages() {
        return gl::genScratchArenas().forSlot(gl::genScratchArenas().slotCount() - 1);
    }

    constexpr int32_t kArity = gl::ExecutionParameters::MAX_ARITY;

    /// @brief An induction step block below the anchor block
    ///        `(AnchorTest91[N,0,s])`, with `main` registered and its
    ///        canonical scope initialized.
    struct GateFixture {
        gl::tests::ProverFixture analyzer{"Peano"};
        gl::Memory root;
        gl::Memory anchor;
        gl::Memory memory;

        GateFixture() {
            analyzer.anchorInfo.name = "AnchorTest91";
            anchor.parentMemory = &root;
            memory.parentMemory = &anchor;
            anchor.setExprKey("(AnchorTest91[N,0,s])");
            memory.level = 2;
            memory.nameMap.encode(sp("main"));
            memory.recursionAuxyImplId = 1;
            analyzer.canonicalizer.initializeCanonicalScope(memory, gl::NameMap::MAIN_ID);
        }
    };

    /// @brief File under @p key at `main` a one-entity instruction with the
    ///        path list @p list (empty for none).
    void fileListed(gl::Memory& m, const std::string& key, const std::string& list) {
        gl::ScratchArena& arena = pages();
        gl::ScratchScope frame(arena);
        gl::WorkInstruction wi(&arena);
        const int32_t mark = wi.elemMark();
        const int32_t arity = static_cast<int32_t>(std::count(key.begin(), key.end(), ',')) + 1;
        wi.commitEntity(sp("atomic"), gl::StrSpan(key), sp(""), arity, mark);
        wi.setMarkedGoal(sp("(goal72[1,2,3])"));
        wi.setPathVariables(gl::StrSpan(list));
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, std::string("main"));
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        entries.findOrEmplaceWork(wi, m.valueInterner);
        entries.writeToCold(m.overallHashMemory.admissionMapIntegration, pk);
    }

    /// @brief The payload values of every instruction under @p key at
    ///        `main`, in stored order.
    std::vector<std::vector<std::string>> payloads(gl::Memory& m, const std::string& key) {
        gl::ScratchArena& arena = pages();
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
} // namespace

// The list text is the sorted, duplicate-free bracketed name set; adding a
// name rebuilds it, a name already present leaves it as is.
TEST(witness_link_gate, path_list_text_is_sorted_and_duplicate_free) {
    gl::ScratchArena& arena = strings();
    gl::ScratchScope frame(arena);
    gl::StrSpan names[4] = { sp("rec"), sp("R"), sp("int_0_lev_6_4"), sp("R") };
    ASSERT_EQ(str(gl::StrSpan(gl::buildPathListScratch(arena, names, 4))), std::string("[R,int_0_lev_6_4,rec]"));
    ASSERT_TRUE(gl::StrSpan(gl::buildPathListScratch(arena, names, 0)).empty());
    ASSERT_EQ(str(gl::StrSpan(gl::extendPathListScratch(arena, sp("[R,rec]"), sp("C")))), std::string("[C,R,rec]"));
    ASSERT_EQ(str(gl::StrSpan(gl::extendPathListScratch(arena, sp("[R,rec]"), sp("rec")))), std::string("[R,rec]"));
    ASSERT_EQ(str(gl::StrSpan(gl::extendPathListScratch(arena, gl::StrSpan(), sp("x")))), std::string("[x]"));
}

// List ids order by their text; the empty list (id 0) comes first.
TEST(witness_link_gate, path_list_ids_order_by_text_with_the_empty_list_first) {
    gl::Memory memory;
    const int32_t longer = memory.valueInterner.encode(sp("[a,b]"));
    const int32_t shorter = memory.valueInterner.encode(sp("[a]"));
    ASSERT_TRUE(gl::pathVariablesLess(0, longer, memory.valueInterner));
    ASSERT_FALSE(gl::pathVariablesLess(longer, 0, memory.valueInterner));
    ASSERT_FALSE(gl::pathVariablesLess(0, 0, memory.valueInterner));
    ASSERT_FALSE(gl::pathVariablesLess(longer, longer, memory.valueInterner));
    ASSERT_TRUE(gl::pathVariablesLess(longer, shorter, memory.valueInterner));
    ASSERT_FALSE(gl::pathVariablesLess(shorter, longer, memory.valueInterner));
}

// The list travels with every copy of an instruction: the integer record,
// the heap record, a load and a cleaned continuation. The empty list mints
// nothing.
TEST(witness_link_gate, instruction_copies_carry_the_path_list) {
    gl::Memory memory;
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    gl::WorkInstruction wi(&arena);
    {
        const int32_t mark = wi.elemMark();
        wi.addElement(sp("(in2[u_1,marker,u_2])"));
        wi.addElement(sp("(in[u_1,u_2])"));
        wi.commitEntity(sp("and"), sp("(and70[u_1,u_2,marker])"), sp(""), 3, mark);
    }
    wi.setMarkedGoal(sp("(goal71[u_1,u_2,marker])"));
    ASSERT_TRUE(wi.pathVariables().empty());
    const gl::IntInstruction unlisted = gl::encodeWorkInstruction(wi, memory.valueInterner);
    ASSERT_EQ(unlisted.pathVariables, 0);

    wi.setPathVariables(sp("[R,rec]"));
    ASSERT_EQ(str(wi.pathVariables()), std::string("[R,rec]"));
    const gl::IntInstruction listed = gl::encodeWorkInstruction(wi, memory.valueInterner);
    ASSERT_EQ(memory.valueInterner.decode(listed.pathVariables), std::string("[R,rec]"));
    gl::WorkInstruction loaded(&arena);
    gl::loadFromIntInstruction(loaded, listed, memory.valueInterner);
    ASSERT_EQ(str(loaded.pathVariables()), std::string("[R,rec]"));
    const gl::Instruction heap = gl::decodeInstruction(listed, memory.valueInterner);
    ASSERT_EQ(heap.pathVariables, std::string("[R,rec]"));
    ASSERT_EQ(gl::encodeInstruction(heap, memory.valueInterner).pathVariables, listed.pathVariables);
    ASSERT_TRUE(gl::decodeInstruction(unlisted, memory.valueInterner).pathVariables.empty());
    ASSERT_FALSE(gl::decodeInstruction(unlisted, memory.valueInterner) == heap);

    gl::WorkInstruction cleaned(&arena);
    gl::cleanInstructionWork(wi, cleaned, sp("(and70[u_1,u_2,marker])"));
    ASSERT_EQ(cleaned.entityCount(), 0);
    ASSERT_EQ(str(cleaned.pathVariables()), std::string("[R,rec]"));

    wi.setPathVariables(gl::StrSpan());
    ASSERT_TRUE(wi.pathVariables().empty());
}

// The stored entry carries the list id through its codec.
TEST(witness_link_gate, integration_entry_codec_carries_the_path_list) {
    gl::IntegrationEntry v;
    v.instruction.markedGoal = 5;
    v.instruction.pathVariables = 9;
    gl::IntLogicalEntity e;
    e.category = 1; e.signature = 2; e.arity = 3; e.definedSet = 4;
    e.elements = { 10, 11 };
    v.instruction.data = { e };
    v.valueIds = { 100 };
    const std::vector<char> bytes = gl::Codec<gl::IntegrationEntry>::serialize(v);
    const gl::IntegrationEntry back = gl::Codec<gl::IntegrationEntry>::deserialize(
        bytes.data(), static_cast<int32_t>(bytes.size()));
    ASSERT_EQ(back.instruction.pathVariables, 9);
    ASSERT_EQ(back.instruction.markedGoal, 5);
    ASSERT_TRUE(back.valueIds == v.valueIds);
    ASSERT_TRUE(gl::Codec<gl::IntegrationEntry>::serialize(back) == bytes);
}

// Instructions differing only in their lists stay separate entries under one
// key, order by list (empty first), and keep list and values through the
// cold store.
TEST(witness_link_gate, integration_map_keeps_path_lists_apart) {
    gl::Memory memory;
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    const auto make = [&](gl::WorkInstruction& wi, const char* list) {
        const int32_t mark = wi.elemMark();
        wi.commitEntity(sp("atomic"), sp("(in2[u_1,marker,u_2])"), sp(""), 3, mark);
        wi.setMarkedGoal(sp("(goal73[u_1,u_2,marker])"));
        wi.setPathVariables(sp(list));
    };
    gl::WorkInstruction a(&arena), b(&arena), c(&arena);
    make(a, "");
    make(b, "[R,rec]");
    make(c, "[R]");
    const int64_t pk = gl::mintTemplateKey(memory.templateInterner, memory.nameMap,
        std::string("(in2[u_1,marker,u_2])"), std::string("main"));
    gl::ArenaIntegrationMap map(&arena, &memory.valueInterner);
    map.buildFromCold(memory.overallHashMemory.admissionMapIntegration, pk);
    ASSERT_EQ(map.findOrEmplaceWork(c, memory.valueInterner), 0);
    ASSERT_EQ(map.findOrEmplaceWork(b, memory.valueInterner), 1);
    ASSERT_EQ(map.findOrEmplaceWork(a, memory.valueInterner), 2);
    ASSERT_EQ(map.findOrEmplaceWork(b, memory.valueInterner), 1);
    ASSERT_EQ(map.entries.size(), 3);
    map.insertValue(1, memory.valueInterner.encode(sp("w")));
    map.writeToCold(memory.overallHashMemory.admissionMapIntegration, pk);

    gl::ArenaIntegrationMap back(&arena, &memory.valueInterner);
    back.buildFromCold(memory.overallHashMemory.admissionMapIntegration, pk);
    ASSERT_EQ(back.entries.size(), 3);
    const char* expected[3] = { "", "[R,rec]", "[R]" };
    for (int32_t i = 0; i < 3; ++i) {
        gl::WorkInstruction loaded(&arena);
        back.loadInstructionInto(i, loaded, memory.valueInterner);
        ASSERT_EQ(str(loaded.pathVariables()), std::string(expected[i]));
        ASSERT_EQ(back.entries[i].valCount, i == 1 ? 1 : 0);
        ASSERT_EQ(back.instructionAt(i).pathVariables, back.entries[i].pathVarsId);
    }
}

// The anchor arguments are those of the first anchor key on the block's
// chain; a prefix of the anchor's name, an empty name or a chain without the
// anchor has none.
TEST(witness_link_gate, anchor_arguments_come_from_the_chain) {
    gl::Memory root, anchor, leaf;
    anchor.parentMemory = &root;
    leaf.parentMemory = &anchor;
    anchor.setExprKey("(AnchorTest91[N,0,s])");
    leaf.setExprKey("(in2[n,m,s])");
    gl::StrSpan out[kArity];
    ASSERT_EQ(gl::ScopeSyntax::anchorArgumentSpans(leaf, sp("AnchorTest91"), out, kArity), 3);
    ASSERT_EQ(str(out[0]), std::string("N"));
    ASSERT_EQ(str(out[1]), std::string("0"));
    ASSERT_EQ(str(out[2]), std::string("s"));
    ASSERT_EQ(gl::ScopeSyntax::anchorArgumentSpans(leaf, sp("AnchorTest9"), out, kArity), 0);
    ASSERT_EQ(gl::ScopeSyntax::anchorArgumentSpans(leaf, gl::StrSpan(), out, kArity), 0);
    ASSERT_EQ(gl::ScopeSyntax::anchorArgumentSpans(root, sp("AnchorTest91"), out, kArity), 0);
}

// Only arguments with the witness's definition set count, typed by position
// (Peano: numbers beside numbers, never the operation or the set); `=` joins
// its arguments and negation changes no type. A part without the witness
// links nothing; a repeated argument counts once.
TEST(witness_link_gate, same_definition_set_arguments_follow_the_typing) {
    gl::tests::ProverFixture analyzer("Peano");
    gl::StrSpan out[kArity];
    ASSERT_EQ(analyzer.scopeTypes.sameDefinitionSetArguments(sp("(in3[w,b,c,+])"), sp("w"), out, kArity), 2);
    ASSERT_EQ(str(out[0]), std::string("b"));
    ASSERT_EQ(str(out[1]), std::string("c"));
    ASSERT_EQ(analyzer.scopeTypes.sameDefinitionSetArguments(sp("(in2[b,w,s])"), sp("w"), out, kArity), 1);
    ASSERT_EQ(str(out[0]), std::string("b"));
    ASSERT_EQ(analyzer.scopeTypes.sameDefinitionSetArguments(sp("!(=[w,r])"), sp("w"), out, kArity), 1);
    ASSERT_EQ(str(out[0]), std::string("r"));
    ASSERT_EQ(analyzer.scopeTypes.sameDefinitionSetArguments(sp("(in[w,N])"), sp("w"), out, kArity), 0);
    ASSERT_EQ(analyzer.scopeTypes.sameDefinitionSetArguments(sp("(in3[a,b,c,+])"), sp("w"), out, kArity), 0);
    ASSERT_EQ(analyzer.scopeTypes.sameDefinitionSetArguments(sp("(in3[b,b,w,+])"), sp("w"), out, kArity), 1);
}

// The linked arguments of a part skip the anchor constants and the
// arguments already collected.
TEST(witness_link_gate, linked_arguments_skip_anchors_and_repeats) {
    GateFixture f;
    auto& gate = f.analyzer.admissionMemory;
    gl::StrSpan out[kArity];
    int32_t n = gate.witnessLinkArguments(f.memory, sp("(in3[w,0,b,+])"), sp("w"), out, 0, kArity);
    ASSERT_EQ(n, 1);
    ASSERT_EQ(str(out[0]), std::string("b"));
    n = gate.witnessLinkArguments(f.memory, sp("(in3[b,c,w,+])"), sp("w"), out, n, kArity);
    ASSERT_EQ(n, 2);
    ASSERT_EQ(str(out[1]), std::string("c"));
    ASSERT_EQ(gate.witnessLinkArguments(f.memory, sp("(in[w,N])"), sp("w"), out, 0, kArity), 0);
}

// A new tree's list holds its fixed arguments without `u_` prefixes and
// without the anchor constants.
TEST(witness_link_gate, root_path_list_drops_prefixes_and_anchors) {
    GateFixture f;
    auto& gate = f.analyzer.admissionMemory;
    gl::ScratchArena& arena = strings();
    gl::ScratchScope frame(arena);
    const gl::StrSpan fixed[4] = { sp("u_rec"), sp("u_N"), sp("R"), sp("u_0") };
    ASSERT_EQ(str(gl::StrSpan(gate.rootPathListScratch(f.memory, fixed, 4, arena))), std::string("[R,rec]"));
    const gl::StrSpan anchorsOnly[2] = { sp("u_N"), sp("s") };
    ASSERT_TRUE(gl::StrSpan(gate.rootPathListScratch(f.memory, anchorsOnly, 2, arena)).empty());
}

// The gate refuses when a linked argument is on the list, only in step
// blocks and only with a list and a linked argument.
TEST(witness_link_gate, path_list_refuses_a_linked_argument_in_step_blocks_only) {
    GateFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const gl::StrSpan linkedR[1] = { sp("R") };
    const gl::StrSpan linkedA[1] = { sp("A") };
    ASSERT_TRUE(gate.witnessLinkRefuses(f.memory, sp("[R,rec]"), linkedR, 1, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[R,rec]"), linkedA, 1, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, gl::StrSpan(), linkedR, 1, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[R,rec]"), nullptr, 0, gl::NameMap::MAIN_ID));
    f.memory.recursionAuxyImplId = 0;
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[R,rec]"), linkedR, 1, gl::NameMap::MAIN_ID));
}

// List and linked arguments compare through the classes of the scope: a
// class member of a list name refuses, and the canonical list holds one
// representative per class, sorted.
TEST(witness_link_gate, refusal_and_lists_compare_through_the_classes) {
    GateFixture f;
    auto& gate = f.analyzer.admissionMemory;
    gl::EquivalenceClass cls;
    cls.setMembersFromNames(std::set<std::string>{ "Q", "R" }, f.memory.nameMap);
    for (auto a : cls.memberIds) for (auto b : cls.memberIds)
        if (a < b) cls.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 2 };
    f.memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(f.memory);
    const gl::StrSpan linkedQ[1] = { sp("Q") };
    const gl::StrSpan linkedA[1] = { sp("A") };
    ASSERT_TRUE(gate.witnessLinkRefuses(f.memory, sp("[R,rec]"), linkedQ, 1, gl::NameMap::MAIN_ID));
    ASSERT_FALSE(gate.witnessLinkRefuses(f.memory, sp("[R,rec]"), linkedA, 1, gl::NameMap::MAIN_ID));

    const std::string representative = str(f.analyzer.canonicalTemplates.canonicalArgumentAtScope(
        f.memory, sp("Q"), gl::NameMap::MAIN_ID));
    ASSERT_TRUE(representative == "Q" || representative == "R");
    ASSERT_EQ(str(f.analyzer.canonicalTemplates.canonicalArgumentAtScope(f.memory, sp("R"), gl::NameMap::MAIN_ID)),
        representative);
    gl::ScratchArena& arena = strings();
    gl::ScratchScope frame(arena);
    ASSERT_EQ(str(gl::StrSpan(gate.canonicalPathListScratch(f.memory, sp("[Q,R,rec]"), gl::NameMap::MAIN_ID, arena))),
        "[" + representative + ",rec]");
    ASSERT_TRUE(gl::StrSpan(gate.canonicalPathListScratch(f.memory, gl::StrSpan(), gl::NameMap::MAIN_ID, arena)).empty());
}

// The route probe counts only instructions the gate leaves open; an
// accepting instruction beside a refusing one opens the route, and outside
// step blocks nothing refuses. A refused commit leaves the key untouched.
TEST(witness_link_gate, route_probe_counts_only_accepting_instructions) {
    GateFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in2[u_2,marker,u_3])";
    fileListed(f.memory, key, "[R]");
    const gl::StrSpan expression = sp("(in2[2,w,3])");
    const gl::StrSpan witness = sp("w");
    const gl::StrSpan marked = sp("(in2[2,marker,3])");
    const gl::StrSpan linkedR[1] = { sp("R") };
    const gl::StrSpan linkedA[1] = { sp("A") };
    ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0, linkedR, 1));
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0, linkedA, 1));
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false));
    ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), true, 0, linkedR, 1));
    ASSERT_EQ(payloads(f.memory, key).size(), std::size_t{1});
    ASSERT_TRUE(payloads(f.memory, key)[0].empty());
    f.memory.recursionAuxyImplId = 0;
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0, linkedR, 1));
    f.memory.recursionAuxyImplId = 1;
    fileListed(f.memory, key, "[S]");
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0, linkedR, 1));
}

// A park records the refusal in the payload of each refusing instruction
// once; the demand probe then no longer counts those instructions for that
// source, while other sources and a new unmarked instruction reopen it.
// Outside step blocks the probe is the plain presence probe.
TEST(witness_link_gate, refusal_marks_close_the_demand_for_their_source_only) {
    GateFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in3[u_2,u_3,marker,u_4])";
    const gl::StrSpan park = sp("(in3[2,3,marker,4])");
    const gl::StrSpan source = sp("(existence93[1,2,3,4,5])");
    const gl::StrSpan other = sp("(existence93[8,2,3,4,5])");
    fileListed(f.memory, key, "[1,9]");
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), source));

    const gl::StrSpan linked[1] = { sp("1") };
    gate.markWitnessLinkRefusals(f.memory, park, sp("main"), linked, 1, source);
    gate.markWitnessLinkRefusals(f.memory, park, sp("main"), linked, 1, source);
    std::vector<std::vector<std::string>> values = payloads(f.memory, key);
    ASSERT_EQ(values.size(), std::size_t{1});
    ASSERT_EQ(values[0].size(), std::size_t{1});
    ASSERT_EQ(values[0][0], std::string("refused:(existence93[1,2,3,4,5])"));
    ASSERT_FALSE(gate.integrationDemandOpen(f.memory, park, sp("main"), source));
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), other));

    const gl::StrSpan offPath[1] = { sp("8") };
    gate.markWitnessLinkRefusals(f.memory, park, sp("main"), offPath, 1, other);
    ASSERT_EQ(payloads(f.memory, key)[0].size(), std::size_t{1});
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), other));

    f.memory.recursionAuxyImplId = 0;
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), source));
    f.memory.recursionAuxyImplId = 1;
    fileListed(f.memory, key, "[9]");
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), source));
}

// The refusal mark is the source in canonical form behind `refused:`.
TEST(witness_link_gate, refusal_mark_is_the_canonical_source) {
    GateFixture f;
    gl::EquivalenceClass cls;
    cls.setMembersFromNames(std::set<std::string>{ "1", "7" }, f.memory.nameMap);
    for (auto a : cls.memberIds) for (auto b : cls.memberIds)
        if (a < b) cls.intEqualityLevelsMap[gl::packEqPairKey(a, b)] = { 2 };
    f.memory.assignClassesById(gl::NameMap::MAIN_ID, { cls });
    seedFixtureCanonicalClasses(f.memory);
    gl::ScratchArena& arena = strings();
    gl::ScratchScope frame(arena);
    const std::string one = str(gl::StrSpan(f.analyzer.admissionMemory.refusalMarkScratch(f.memory,
        sp("(existence93[1,2,3,4,5])"), gl::NameMap::MAIN_ID, arena)));
    const std::string seven = str(gl::StrSpan(f.analyzer.admissionMemory.refusalMarkScratch(f.memory,
        sp("(existence93[7,2,3,4,5])"), gl::NameMap::MAIN_ID, arena)));
    ASSERT_EQ(one, seven);
    ASSERT_TRUE(one == "refused:(existence93[1,2,3,4,5])" || one == "refused:(existence93[7,2,3,4,5])");
}
