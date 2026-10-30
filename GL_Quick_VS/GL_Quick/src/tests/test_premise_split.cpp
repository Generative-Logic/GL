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
/// @brief Unit tests for splitting identical hypothesis integration goals by
///        premise (D-579).
///
/// @details
/// When one weak induction hypothesis install prepares N identical
/// integration goals, every instruction of that preparation is filed N times.
/// Copy i carries two forbidden sets computed from the original premises (the
/// decimal variables and the logic-block levels that belong only to the
/// group's other premises) and refuses an arriving expression whose
/// signature names one of them. Covers the set helpers, the storage of the
/// sets in every instruction form, the install on FTA2 row 20, the route-3
/// probe and commit, the park marks and the statement fill. A copy also
/// carries its own premise's level; a first fill of a copy at `main` is gated
/// on an arriving expression of numbers and opens the copy's int scope
/// (D-578, `test_int_scope.cpp`).

#include "../memory.hpp"
#include "test_harness.hpp"
#include "prover_fixture.hpp"
#include "../logical_block/derivation_tree.hpp"
#include "../prover.hpp"

#include <cstring>
#include <map>
#include <set>
#include <string>
#include <utility>
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

    /// @brief The mask holding one member.
    uint64_t bit(int member) { return gl::forbiddenSetBit(member); }

    /// @brief The split binding of a copy (tests only).
    gl::IntegrationForbiddenSets sets(uint64_t variables, uint64_t levels, int32_t premiseLevel) {
        gl::IntegrationForbiddenSets out;
        out.variables = variables;
        out.levels = levels;
        out.premiseLevel = premiseLevel;
        return out;
    }

    /// @brief Row 20's copy bound to P (level 1): forbids Q (12) and Q's
    ///        block level 2.
    const gl::IntegrationForbiddenSets kCopyP = sets(uint64_t{1} << 12, uint64_t{1} << 2, 1);
    /// @brief Row 20's copy bound to Q (level 2): forbids P (10) and P's
    ///        block level 1.
    const gl::IntegrationForbiddenSets kCopyQ = sets(uint64_t{1} << 10, uint64_t{1} << 1, 2);

    /// @brief An induction step block (level 3) below the anchor block
    ///        `(AnchorTest92[N,0,s])` and the blocks of row 20's premises P
    ///        (level 1) and Q (level 2), whose hypothesis install split
    ///        identical goals, with `main` registered and its canonical
    ///        scope initialized.
    struct SplitFixture {
        gl::tests::ProverFixture analyzer{"Peano"};
        gl::Memory root;
        gl::Memory anchor;
        gl::Memory premiseP;
        gl::Memory premiseQ;
        gl::Memory memory;

        SplitFixture() {
            analyzer.anchorInfo.name = "AnchorTest92";
            anchor.parentMemory = &root;
            premiseP.parentMemory = &anchor;
            premiseQ.parentMemory = &premiseP;
            memory.parentMemory = &premiseQ;
            anchor.setExprKey("(AnchorTest92[N,0,s])");
            premiseP.setExprKey("(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])");
            premiseQ.setExprKey("(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])");
            premiseP.level = 1;
            premiseQ.level = 2;
            memory.level = 3;
            memory.nameMap.encode(sp("main"));
            memory.recursionAuxyImplId = 1;
            memory.premiseSplitFiled = true;
            analyzer.canonicalizer.initializeCanonicalScope(memory, gl::NameMap::MAIN_ID);
        }
    };

    /// @brief File under @p key at `main` one copy per forbidden-set pair of a
    ///        one-entity instruction, as the split install does.
    void fileCopies(gl::Memory& m, const std::string& key,
                    const std::vector<gl::IntegrationForbiddenSets>& copies) {
        gl::ScratchArena& arena = pages();
        gl::ScratchScope frame(arena);
        gl::WorkInstruction wi(&arena);
        const int32_t mark = wi.elemMark();
        int32_t arity = 1;
        for (const char c : key) arity += c == ',' ? 1 : 0;
        wi.commitEntity(sp("atomic"), gl::StrSpan(key), sp(""), arity, mark);
        wi.setMarkedGoal(sp("(goal74[1,2,3])"));
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, std::string("main"));
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        for (const gl::IntegrationForbiddenSets& copy : copies) {
            wi.forbidden = copy;
            entries.findOrEmplaceWork(wi, m.valueInterner);
        }
        entries.writeToCold(m.overallHashMemory.admissionMapIntegration, pk);
        m.lbMemory.canonicalUses.integrationDemands.changed(pk);
    }

    /// @brief The payload values of every instruction under @p key at
    ///        `main`, keyed by the copy's forbidden variables.
    std::map<uint64_t, std::vector<std::string>> valuesByCopy(gl::Memory& m, const std::string& key) {
        gl::ScratchArena& arena = pages();
        gl::ScratchScope frame(arena);
        const int64_t pk = gl::mintTemplateKey(m.templateInterner, m.nameMap, key, std::string("main"));
        gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
        entries.buildFromCold(m.overallHashMemory.admissionMapIntegration, pk);
        std::map<uint64_t, std::vector<std::string>> out;
        for (int32_t i = 0; i < entries.entries.size(); ++i) {
            int32_t ids[gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES];
            const int32_t n = entries.valuesAt(i, ids, gl::ExecutionParameters::MAX_INTEGRATION_ENTRY_VALUES);
            std::vector<std::string> values;
            for (int32_t v = 0; v < n; ++v) values.push_back(m.valueInterner.decode(ids[v]));
            out[entries.entries[i].forbidden.variables] = values;
        }
        return out;
    }

    /// @brief An arriving-expression probe whose start is @p arriving.
    gl::WitnessHistoryProbe arrival(const char* arriving) {
        gl::WitnessHistoryProbe probe;
        probe.start = sp(arriving);
        probe.startScope = sp("main");
        return probe;
    }
} // namespace

// A member is one bit of a 64-bit mask; values outside [0, 64) are never
// members.
TEST(premise_split, forbidden_set_bits_and_membership) {
    ASSERT_EQ(gl::forbiddenSetBit(0), uint64_t{1});
    ASSERT_EQ(gl::forbiddenSetBit(12), uint64_t{1} << 12);
    ASSERT_EQ(gl::forbiddenSetBit(63), uint64_t{1} << 63);
    const uint64_t set = bit(2) | bit(63);
    ASSERT_TRUE(gl::forbiddenSetHas(set, 2));
    ASSERT_TRUE(gl::forbiddenSetHas(set, 63));
    ASSERT_FALSE(gl::forbiddenSetHas(set, 3));
    ASSERT_FALSE(gl::forbiddenSetHas(set, 64));
    ASSERT_FALSE(gl::forbiddenSetHas(set, -1));
}

// Row 20: the copy bound to P forbids Q's own variable and level, the copy
// bound to Q the reverse. With three premises a copy forbids every variable
// of the others that its own premise lacks, shared ones included.
TEST(premise_split, premise_sets_bind_each_copy_to_its_premise) {
    const gl::StrSpan row20[2] = { sp("(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"),
                                   sp("(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])") };
    const int32_t levels[2] = { 1, 2 };
    uint64_t variables = 0;
    uint64_t forbiddenLevels = 0;
    gl::premiseForbiddenSets(row20, levels, 2, 0, variables, forbiddenLevels);
    ASSERT_EQ(variables, kCopyP.variables);
    ASSERT_EQ(forbiddenLevels, kCopyP.levels);
    gl::premiseForbiddenSets(row20, levels, 2, 1, variables, forbiddenLevels);
    ASSERT_EQ(variables, kCopyQ.variables);
    ASSERT_EQ(forbiddenLevels, kCopyQ.levels);

    const gl::StrSpan three[3] = { sp("(f[1,2,3])"), sp("!(f[1,4,5])"), sp("(f[1,4,6])") };
    const int32_t threeLevels[3] = { 1, 2, 3 };
    gl::premiseForbiddenSets(three, threeLevels, 3, 0, variables, forbiddenLevels);
    ASSERT_EQ(variables, bit(4) | bit(5) | bit(6));
    ASSERT_EQ(forbiddenLevels, bit(2) | bit(3));
    gl::premiseForbiddenSets(three, threeLevels, 3, 1, variables, forbiddenLevels);
    ASSERT_EQ(variables, bit(2) | bit(3) | bit(6));
    ASSERT_EQ(forbiddenLevels, bit(1) | bit(3));
}

// The signature names decimals and forced-witness levels; `rec`, other
// witness classes, anchor copies and values of 64 or more name nothing. A
// copy refuses an expression that names a member of its sets.
TEST(premise_split, signature_tokens_and_cross_hits) {
    uint64_t variables = 0;
    uint64_t levels = 0;
    gl::signatureTokenSets(sp("(existence12[1,4,rec,force_lev_2_2,2])"), variables, levels);
    ASSERT_EQ(variables, bit(1) | bit(4) | bit(2));
    ASSERT_EQ(levels, bit(2));
    gl::signatureTokenSets(sp("!(existence12[u_1,u_12,int_0_lev_3_0,it_1_lev_2_5,x2,70])"), variables, levels);
    ASSERT_EQ(variables, bit(1) | bit(12));
    ASSERT_EQ(levels, uint64_t{0});

    const auto hits = [](const char* expression, const gl::IntegrationForbiddenSets& copy) {
        return gl::argumentsHitForbiddenSets(sp(expression), copy.variables, copy.levels);
    };
    ASSERT_TRUE(hits("(existence12[1,4,rec,12,2])", kCopyP));
    ASSERT_FALSE(hits("(existence12[1,4,rec,12,2])", kCopyQ));
    ASSERT_TRUE(hits("(existence12[1,4,rec,force_lev_2_2,2])", kCopyP));
    ASSERT_FALSE(hits("(existence12[1,4,rec,force_lev_2_2,2])", kCopyQ));
    ASSERT_FALSE(hits("(existence12[1,4,rec,10,2])", kCopyP));
    ASSERT_TRUE(hits("(existence12[1,4,rec,10,2])", kCopyQ));
    ASSERT_FALSE(hits("(existence39[2,11,1,4])", kCopyP));
    ASSERT_FALSE(hits("(existence39[2,11,1,4])", kCopyQ));
    ASSERT_TRUE(hits("(in2[u_12,x,3])", kCopyP));
    ASSERT_FALSE(hits("(existence12[1,4,rec,12,2])", gl::IntegrationForbiddenSets{}));
}

// Each set travels as two int32 words, low half first; the order compares
// variables, then levels, then the premise level.
TEST(premise_split, forbidden_set_words_and_order) {
    const uint64_t set = bit(63) | bit(40) | bit(12) | bit(0);
    ASSERT_EQ(gl::forbiddenSetFromWords(gl::forbiddenSetWord(set, false), gl::forbiddenSetWord(set, true)), set);
    ASSERT_EQ(gl::forbiddenSetWord(bit(33), false), 0);
    ASSERT_EQ(gl::forbiddenSetWord(bit(33), true), 2);
    const gl::IntegrationForbiddenSets none;
    ASSERT_EQ(gl::compareForbiddenSets(none, none), 0);
    ASSERT_TRUE(gl::compareForbiddenSets(none, kCopyP) < 0);
    ASSERT_TRUE(gl::compareForbiddenSets(kCopyQ, kCopyP) < 0);
    ASSERT_TRUE(gl::compareForbiddenSets(kCopyP, kCopyQ) > 0);
    ASSERT_TRUE(gl::compareForbiddenSets(kCopyP, sets(kCopyP.variables, bit(3), 1)) < 0);
    ASSERT_TRUE(gl::compareForbiddenSets(kCopyP, sets(kCopyP.variables, kCopyP.levels, 2)) < 0);
    ASSERT_EQ(gl::compareForbiddenSets(kCopyP, sets(kCopyP.variables, kCopyP.levels, 1)), 0);
}

// The sets travel with every form and copy of an instruction: the integer
// record, the heap record, a load, a cleaned continuation and the codec.
TEST(premise_split, instruction_forms_carry_the_sets) {
    gl::Memory memory;
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    gl::WorkInstruction wi(&arena);
    {
        const int32_t mark = wi.elemMark();
        wi.addElement(sp("(in2[u_1,marker,u_2])"));
        wi.commitEntity(sp("and"), sp("(and75[u_1,u_2,marker])"), sp(""), 3, mark);
    }
    wi.setMarkedGoal(sp("(goal75[u_1,u_2,marker])"));
    wi.forbidden = kCopyP;
    const gl::IntInstruction stored = gl::encodeWorkInstruction(wi, memory.valueInterner);
    ASSERT_EQ(gl::compareForbiddenSets(stored.forbidden, kCopyP), 0);
    gl::WorkInstruction loaded(&arena);
    gl::loadFromIntInstruction(loaded, stored, memory.valueInterner);
    ASSERT_EQ(gl::compareForbiddenSets(loaded.forbidden, kCopyP), 0);
    const gl::Instruction heap = gl::decodeInstruction(stored, memory.valueInterner);
    ASSERT_EQ(gl::compareForbiddenSets(heap.forbidden, kCopyP), 0);
    ASSERT_EQ(gl::compareForbiddenSets(gl::encodeInstruction(heap, memory.valueInterner).forbidden, kCopyP), 0);
    gl::Instruction other = heap;
    other.forbidden = kCopyQ;
    ASSERT_FALSE(other == heap);
    ASSERT_TRUE(other < heap);
    ASSERT_FALSE(heap < other);

    gl::WorkInstruction cleaned(&arena);
    gl::cleanInstructionWork(wi, cleaned, sp("(and75[u_1,u_2,marker])"));
    ASSERT_EQ(cleaned.entityCount(), 0);
    ASSERT_EQ(gl::compareForbiddenSets(cleaned.forbidden, kCopyP), 0);

    gl::IntegrationEntry entry;
    entry.instruction = stored;
    entry.valueIds = { 7 };
    const std::vector<char> bytes = gl::Codec<gl::IntegrationEntry>::serialize(entry);
    const gl::IntegrationEntry back = gl::Codec<gl::IntegrationEntry>::deserialize(
        bytes.data(), static_cast<int32_t>(bytes.size()));
    ASSERT_EQ(gl::compareForbiddenSets(back.instruction.forbidden, kCopyP), 0);
    ASSERT_TRUE(gl::Codec<gl::IntegrationEntry>::serialize(back) == bytes);
}

// Copies differing only in their sets stay separate entries under one key,
// order by their sets (unsplit first) and keep them through the cold store,
// whose bytes are the codec's.
TEST(premise_split, integration_map_keeps_split_copies_apart) {
    gl::Memory memory;
    gl::ScratchArena& arena = pages();
    gl::ScratchScope frame(arena);
    gl::WorkInstruction wi(&arena);
    {
        const int32_t mark = wi.elemMark();
        wi.commitEntity(sp("atomic"), sp("(in2[u_1,marker,u_2])"), sp(""), 3, mark);
    }
    wi.setMarkedGoal(sp("(goal76[u_1,u_2,marker])"));
    const int64_t pk = gl::mintTemplateKey(memory.templateInterner, memory.nameMap,
        std::string("(in2[u_1,marker,u_2])"), std::string("main"));
    gl::ArenaIntegrationMap map(&arena, &memory.valueInterner);
    map.buildFromCold(memory.overallHashMemory.admissionMapIntegration, pk);
    wi.forbidden = kCopyP;
    ASSERT_EQ(map.findOrEmplaceWork(wi, memory.valueInterner), 0);
    wi.forbidden = gl::IntegrationForbiddenSets{};
    ASSERT_EQ(map.findOrEmplaceWork(wi, memory.valueInterner), 1);
    wi.forbidden = kCopyQ;
    ASSERT_EQ(map.findOrEmplaceWork(wi, memory.valueInterner), 2);
    wi.forbidden = kCopyP;
    ASSERT_EQ(map.findOrEmplaceWork(wi, memory.valueInterner), 0);
    ASSERT_EQ(map.entries.size(), 3);
    map.insertValue(0, memory.valueInterner.encode(sp("w")));
    const std::vector<gl::IntegrationEntry> heapRun = map.flatten();
    map.writeToCold(memory.overallHashMemory.admissionMapIntegration, pk);

    auto& cold = memory.overallHashMemory.admissionMapIntegration;
    ASSERT_EQ(cold.runLen(cold.lookup(pk)), 3);
    for (int32_t i = 0; i < 3; ++i)
        ASSERT_TRUE(gl::Codec<gl::IntegrationEntry>::serialize(heapRun[static_cast<std::size_t>(i)])
            == gl::Codec<gl::IntegrationEntry>::serialize(cold.recordAt(cold.lookup(pk), i)));
    gl::ArenaIntegrationMap back(&arena, &memory.valueInterner);
    back.buildFromCold(cold, pk);
    ASSERT_EQ(back.entries.size(), 3);
    const gl::IntegrationForbiddenSets expected[3] = { gl::IntegrationForbiddenSets{}, kCopyQ, kCopyP };
    for (int32_t i = 0; i < 3; ++i) {
        ASSERT_EQ(gl::compareForbiddenSets(back.entries[i].forbidden, expected[i]), 0);
        ASSERT_EQ(gl::compareForbiddenSets(back.instructionAt(i).forbidden, expected[i]), 0);
        gl::WorkInstruction loaded(&arena);
        back.loadInstructionInto(i, loaded, memory.valueInterner);
        ASSERT_EQ(gl::compareForbiddenSets(loaded.forbidden, expected[i]), 0);
        ASSERT_EQ(back.entries[i].valCount, i == 2 ? 1 : 0);
    }
}

// Row 20's weak hypothesis install prepares the two-marker goal once and
// files each of its instructions twice, one copy bound to P and one to Q.
// Row 20 builds no strong step block: its length sits in an operator with a
// numeric output (D-594).
TEST(premise_split, weak_install_files_each_identical_goal_once_per_premise) {
    gl::tests::ProverFixture ea(std::string("FTA"));
    const std::string row20 =
        "(>[1,2,3,4,5,6,7,8](AnchorFTA[1,2,3,4,5,6,7,8])(>[9,10,11](primeFactorSeq[1,3,4,5,2,6,7,9,10,11])"
        "(>[12](primeFactorSeq[1,3,4,5,2,6,7,9,12,11])(=[10,12]))))";
    ea.proofGridBuilder.addTheoremToMemory(row20, ea.body, 0, false, ea.globalDependencies);
    gl::Memory* anchorLB = ea.simpleMapStore.findChild(&ea.body, "(AnchorFTA[1,2,3,4,5,6,7,8])");
    ASSERT_TRUE(anchorLB != nullptr);
    gl::Memory* firstLB = ea.simpleMapStore.findChild(anchorLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])");
    ASSERT_TRUE(firstLB != nullptr);
    gl::Memory* goalLB = ea.simpleMapStore.findChild(firstLB, "(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])");
    ASSERT_TRUE(goalLB != nullptr);
    gl::Memory* weak = ea.simpleMapStore.findChild(goalLB, "(in2[rec0,9,3])");
    gl::Memory* strong = ea.simpleMapStore.findChild(goalLB, "(in2[rec1,9,3])");
    ASSERT_TRUE(weak != nullptr && strong == nullptr);
    ASSERT_TRUE(weak->premiseSplitFiled);
    ASSERT_TRUE(weak->weakStepBlock);
    ASSERT_FALSE(weak->strongStepBlock);

    // Every integration key of the weak step block holds `rec`, the
    // induction variable 9 or a size marker
    // (D-583).
    {
        const auto& keys = weak->overallHashMemory.admissionMapIntegration;
        int32_t filed = 0;
        for (int32_t id = 1; id <= keys.count(); ++id) {
            if (keys.runLen(id) == 0) continue;
            const gl::StrSpan text = weak->templateInterner.decodeView(
                static_cast<int32_t>(static_cast<uint64_t>(keys.keyAt(id)) >> 32));
            gl::StrSpan args[gl::ExecutionParameters::MAX_ARITY];
            const int32_t argN = gl::getArgsSpans(text, args, gl::ExecutionParameters::MAX_ARITY);
            bool connected = false;
            for (int32_t a = 0; a < argN && !connected; ++a) {
                const gl::StrSpan argument = gl::startsWithSpan(args[a], "u_", 2)
                    ? gl::StrSpan(args[a].ptr + 2, args[a].len - 2) : args[a];
                const int32_t value = weak->valueInterner.lookup(argument);
                connected = gl::equalSpans(argument, sp("rec")) || gl::equalSpans(argument, sp("9"))
                    || (value != 0 && weak->lbMemory.sizeMarkers.lookup(value) != 0);
            }
            ASSERT_TRUE(connected);
            ++filed;
        }
        ASSERT_TRUE(filed > 0);
    }

    // Per two-marker label, the forbidden sets of its copies over all keys;
    // every key holding a split copy holds `rec` or the induction variable 9
    // (D-578), and each copy carries its premise's
    // level.
    const auto copiesByLabel = [](gl::Memory& m) {
        std::map<std::string, std::multiset<std::pair<uint64_t, uint64_t>>> out;
        int32_t unsplitTwoMarker = 0;
        const auto& keys = m.overallHashMemory.admissionMapIntegration;
        gl::ScratchArena& arena = pages();
        for (int32_t id = 1; id <= keys.count(); ++id) {
            if (keys.lookup(keys.keyAt(id)) == 0) continue;
            gl::ScratchScope frame(arena);
            gl::ArenaIntegrationMap entries(&arena, &m.valueInterner);
            entries.buildFromCold(keys, keys.keyAt(id));
            for (int32_t i = 0; i < entries.entries.size(); ++i) {
                gl::WorkInstruction wi(&arena);
                entries.loadInstructionInto(i, wi, m.valueInterner);
                const std::string label = wi.markedGoal().toStdString();
                if (label.find("marker1") == std::string::npos) continue;
                if (wi.forbidden.variables == 0 && wi.forbidden.levels == 0) ++unsplitTwoMarker;
                if (wi.forbidden.premiseLevel != 0) {
                    const gl::StrSpan key = m.templateInterner.decodeView(
                        static_cast<int32_t>(static_cast<uint64_t>(keys.keyAt(id)) >> 32));
                    const std::string text = key.toStdString();
                    ASSERT_TRUE(text.find("u_rec") != std::string::npos || text.find("u_9,") != std::string::npos
                        || text.find("u_9]") != std::string::npos);
                    ASSERT_EQ(wi.forbidden.premiseLevel, wi.forbidden.levels == kCopyP.levels ? 1 : 2);
                }
                out[label + "@" + std::to_string(id)].insert({ wi.forbidden.variables, wi.forbidden.levels });
            }
        }
        return std::make_pair(out, unsplitTwoMarker);
    };
    const auto weakCopies = copiesByLabel(*weak);
    ASSERT_FALSE(weakCopies.first.empty());
    ASSERT_EQ(weakCopies.second, 0);
    const std::pair<uint64_t, uint64_t> boundToP(kCopyP.variables, kCopyP.levels);
    const std::pair<uint64_t, uint64_t> boundToQ(kCopyQ.variables, kCopyQ.levels);
    for (const auto& [label, copies] : weakCopies.first) {
        const std::size_t pCount = copies.count(boundToP);
        ASSERT_TRUE(pCount > 0);
        ASSERT_EQ(copies.count(boundToQ), pCount);
        ASSERT_EQ(copies.size(), 2 * pCount);
    }
}

// The route probe and its commit read each copy's sets against the arriving
// expression: P's prefix reaches only the copy bound to P, Q's only the copy
// bound to Q, an expression naming both reaches none. At `main` a root copy
// also refuses an arriving expression with an argument that is not a number,
// and a commit opens the accepting copy's int scope. Without the block's
// split flag the probe is the plain presence probe.
TEST(premise_split, route_probe_and_commit_refuse_cross_arrivals) {
    SplitFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in2[u_2,marker,u_3])";
    fileCopies(f.memory, key, { kCopyP, kCopyQ });
    const gl::StrSpan expression = sp("(in2[2,w,3])");
    const gl::StrSpan witness = sp("w");
    const gl::StrSpan marked = sp("(in2[2,marker,3])");

    const gl::WitnessHistoryProbe both = arrival("(existence12[1,4,rec,10,12])");
    gl::InductionScopeSink opened;
    ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0,
        nullptr, 0, &both));
    ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), true, 0,
        nullptr, 0, &both, &opened));
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyP.variables].empty());
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyQ.variables].empty());
    ASSERT_EQ(opened.count, 0);

    const gl::WitnessHistoryProbe fromQ = arrival("(existence12[1,4,rec,12,2])");
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0,
        nullptr, 0, &fromQ));
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), true, 0,
        nullptr, 0, &fromQ, &opened));
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyP.variables].empty());
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyQ.variables] == std::vector<std::string>{ "w" });
    ASSERT_EQ(opened.count, 1);
    ASSERT_FALSE(opened.publishedHere);
    ASSERT_EQ(f.memory.nameMap.decode(opened.scopes[0]),
        std::string("main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])_w"));

    const gl::WitnessHistoryProbe forcedQ = arrival("(existence12[1,4,rec,force_lev_2_2,2])");
    gl::InductionScopeSink none;
    ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,v,3])"), sp("v"), marked, sp("main"), false, 0,
        nullptr, 0, &forcedQ));
    ASSERT_FALSE(gate.isAdmittedIntegration(f.memory, sp("(in2[2,v,3])"), sp("v"), marked, sp("main"), true, 0,
        nullptr, 0, &forcedQ, &none));
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyP.variables].empty());
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyQ.variables] == std::vector<std::string>{ "w" });
    ASSERT_EQ(none.count, 0);

    f.memory.premiseSplitFiled = false;
    ASSERT_TRUE(gate.isAdmittedIntegration(f.memory, expression, witness, marked, sp("main"), false, 0,
        nullptr, 0, &both));
}

// A park records the refusal mark in the copies whose sets the parked source
// hits; the demand stays open while one copy is unmarked.
TEST(premise_split, park_marks_close_the_demand_of_refusing_copies_only) {
    SplitFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in3[u_2,u_3,marker,u_4])";
    const gl::StrSpan park = sp("(in3[2,3,marker,4])");
    fileCopies(f.memory, key, { kCopyP, kCopyQ });
    const gl::StrSpan fromQ = sp("(existence93[1,2,3,4,12])");
    gate.markWitnessLinkRefusals(f.memory, park, sp("main"), nullptr, 0, fromQ, nullptr);
    std::map<uint64_t, std::vector<std::string>> values = valuesByCopy(f.memory, key);
    ASSERT_TRUE(values[kCopyP.variables] == std::vector<std::string>{ "refused:(existence93[1,2,3,4,12])" });
    ASSERT_TRUE(values[kCopyQ.variables].empty());
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), fromQ));

    const gl::StrSpan fromBoth = sp("(existence93[1,2,3,10,12])");
    gate.markWitnessLinkRefusals(f.memory, park, sp("main"), nullptr, 0, fromBoth, nullptr);
    ASSERT_FALSE(gate.integrationDemandOpen(f.memory, park, sp("main"), fromBoth));
    ASSERT_TRUE(gate.integrationDemandOpen(f.memory, park, sp("main"), fromQ));

    // A source with an argument that is not a number fails both root copies'
    // initiation gate at `main`.
    const gl::StrSpan witnessArgument = sp("(existence93[1,2,3,int_0_lev_3_0,4])");
    gate.markWitnessLinkRefusals(f.memory, park, sp("main"), nullptr, 0, witnessArgument, nullptr);
    ASSERT_FALSE(gate.integrationDemandOpen(f.memory, park, sp("main"), witnessArgument));
}

// The statement fill refuses a copy whose sets the arriving expression's
// signature hits and fills the others; at `main` a root copy also refuses an
// arriving expression with an argument that is not a number. Without
// arriving tokens every copy fills, each opening its int scope.
TEST(premise_split, statement_fill_refuses_cross_arrivals) {
    SplitFixture f;
    auto& gate = f.analyzer.admissionMemory;
    const std::string key = "(in2[u_2,marker,u_3])";
    fileCopies(f.memory, key, { kCopyP, kCopyQ });
    uint64_t variables = 0;
    uint64_t levels = 0;

    gl::signatureTokenSets(sp("(existence12[1,4,rec,12,2])"), variables, levels);
    gate.updateAdmissionMapIntegration(sp("(in2[2,w,3])"), f.memory, sp("main"), variables, levels, true);
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyP.variables].empty());
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyQ.variables] == std::vector<std::string>{ "w" });

    gl::signatureTokenSets(sp("(existence12[1,4,rec,force_lev_1_2,2])"), variables, levels);
    gate.updateAdmissionMapIntegration(sp("(in2[2,u,3])"), f.memory, sp("main"), variables, levels,
        gl::hasOnlyNumberFreeArguments(sp("(existence12[1,4,rec,force_lev_1_2,2])")));
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyP.variables].empty());
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyQ.variables] == std::vector<std::string>{ "w" });

    gate.updateAdmissionMapIntegration(sp("(in2[2,t,3])"), f.memory, sp("main"), 0, 0, true);
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyP.variables] == std::vector<std::string>{ "t" });
    ASSERT_TRUE(valuesByCopy(f.memory, key)[kCopyQ.variables] == (std::vector<std::string>{ "t", "w" }));
    ASSERT_TRUE(f.memory.nameMap.lookup(std::string(
        "main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,10,11])_t")) != 0);
    ASSERT_TRUE(f.memory.nameMap.lookup(std::string(
        "main_boundary_int_scope_(primeFactorSeq[1,3,4,5,2,6,7,9,12,11])_t")) != 0);
}
