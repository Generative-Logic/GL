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
#include "isolated_test_directory.hpp"
#include "hashburst_dump_expected.hpp"
#include "../infra/hashburst_dump.hpp"
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

TEST(hashburst_dump, preserves_entry_exit_order_and_first_entry_definition_section) {
    gl::tests::IsolatedTestDirectory directory("hashburst_dump_test");
    gl::Memory root;
    gl::Memory block;
    ((block).parentMemory) = &root;
    block.setExprKey("(fixture[u_1])");
    ((block).level) = 0;
    ((block).startInt) = 7;
    const gl::EncodedExpression statement("(p[a])", "main");
    ((block).intEncodedStatements).push_back(gl::encodeExpression(statement, ((block).nameMap)));
    const auto goal = ((block).nameMap).encode("(q[a])");
    ((block).intToBeProved).insertSorted(gl::packStatementKey(goal, gl::NameMap::MAIN_ID), 0);
    gl::CompiledExpressionMap compiled;
    compiled["p"] = gl::LogicalEntity("atomic", {""}, "(p[u_1])", 1);
    compiled["q"] = gl::LogicalEntity("atomic", {""}, "(q[u_1])", 1);
    compiled["orFixture"] = gl::LogicalEntity("or", {"(p[u_1])", "(q[u_1])"}, "(orFixture[u_1])", 1);
    ASSERT_FALSE(gl::hashburst_dump::isTargetLB(block));
    std::filesystem::create_directories(".debug");
    gl::hashburst_dump::dumpEntry(block, compiled);
    gl::hashburst_dump::dumpExit(block);
    ((block).startInt) = 9;
    ((block).intEncodedStatements).push_back(gl::encodeExpression(
        gl::EncodedExpression("!(q[a])", "main"), ((block).nameMap)));
    gl::hashburst_dump::dumpEntry(block, compiled);
    gl::hashburst_dump::dumpExit(block);
    std::ifstream input(".debug/hashburst_trace.txt");
    ASSERT_TRUE(input.is_open());
    const std::string dump((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    std::string expected;
    for (const char* part : gl::tests::expectedHashburstDumpParts) expected += part;
    ASSERT_EQ(dump, expected);
    const auto firstEntry = dump.find("=== HASHBURST ENTRY #1");
    const auto firstExit = dump.find("=== HASHBURST EXIT #1");
    const auto secondEntry = dump.find("=== HASHBURST ENTRY #2");
    const auto secondExit = dump.find("=== HASHBURST EXIT #2");
    ASSERT_TRUE(secondExit != std::string::npos);
    ASSERT_TRUE(firstEntry != std::string::npos && firstEntry < firstExit);
    ASSERT_TRUE(firstExit < secondEntry && secondEntry < secondExit);
    const auto definitions = dump.find("-- compiledExpressions");
    ASSERT_TRUE(definitions != std::string::npos && definitions < firstExit);
    ASSERT_TRUE(dump.find("-- compiledExpressions", definitions + 1) == std::string::npos);
    ASSERT_TRUE(dump.find("-- counters: startInt=7") != std::string::npos);
    ASSERT_TRUE(dump.find("-- counters: startInt=9", secondEntry) != std::string::npos);
    ASSERT_TRUE(dump.find("(p[a]) | v=main") != std::string::npos);
    ASSERT_TRUE(dump.find("(q[a]) | v=main") != std::string::npos);
}

// recordPhase2Upload buffers the target block's upload: the rows sent to the
// device and the rows with a premise core that the premise patterns held back
// (rows without a premise core are only counted); writePhase2Uploads writes
// them as the EXIT section and empties the buffer, so the next section is the
// zero-upload header.
TEST(hashburst_dump, phase2_upload_section_lists_sent_and_withheld_rows) {
    gl::tests::IsolatedTestDirectory directory("hashburst_dump_upload_test");
    gl::Memory block;
    gl::NameMap& names = ((block).nameMap);
    for (const char* text : { "(p[a])", "(p[b])", "(q[c])", "!(p[d])" }) {
        ((block).intEncodedStatements).push_back(gl::encodeExpression(
            gl::EncodedExpression(text, "main"), names));
    }
    // Only the positive p core is a premise core.
    std::vector<uint8_t> coreUsed(static_cast<std::size_t>(names.nameCount()) + 1, 0);
    coreUsed[static_cast<std::size_t>(names.lookup(gl::StrSpan("p", 1)))] = 1;
    const gl::NameId sent[1] = { 1 };
    gl::hashburst_dump::recordPhase2Upload(block, sent, 1, coreUsed.data(), coreUsed.size());
    std::filesystem::create_directories(".debug");
    {
        std::ofstream f(".debug/upload_section.txt");
        ASSERT_TRUE(f.is_open());
        gl::hashburst_dump::writePhase2Uploads(f);
        gl::hashburst_dump::writePhase2Uploads(f);
    }
    std::ifstream input(".debug/upload_section.txt");
    ASSERT_TRUE(input.is_open());
    const std::string section((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_EQ(section, std::string(
        "-- phase2.statementUpload (1 uploads, 1 sent, 1 withheld by the premise patterns):\n"
        "  upload #1: 4 rows, 1 sent, 1 withheld, 2 without a premise core\n"
        "  withheld [0] (p[a]) | v=main\n"
        "  sent [1] (p[b]) | v=main\n"
        "-- phase2.statementUpload (0 uploads, 0 sent, 0 withheld by the premise patterns):\n"));
}

// An admission-integration instruction line ends with the instruction's path
// list when it has one; an instruction without a list keeps the plain line
// (D-529).
TEST(hashburst_dump, integration_instruction_line_carries_its_path_list) {
    gl::tests::IsolatedTestDirectory directory("hashburst_dump_path_test");
    gl::Memory root;
    gl::Memory block;
    ((block).parentMemory) = &root;
    block.setExprKey("(fixture[u_1])");
    gl::ValueInterner& values = ((block).valueInterner);
    const int64_t pk = gl::mintTemplateKey(((block).templateInterner), ((block).nameMap),
        "(in2[u_a,marker,3])", "main");
    gl::IntInstruction listed;
    gl::IntLogicalEntity entity;
    entity.category = values.encode("atomic");
    entity.signature = values.encode("(in2[u_a,marker,3])");
    entity.elements = { values.encode("u_a"), values.encode("marker"), values.encode("3") };
    entity.arity = 3;
    entity.definedSet = values.encode("N");
    listed.data.push_back(entity);
    listed.markedGoal = values.encode("(fold[a,marker])");
    listed.pathVariables = values.encode("[R,rec]");
    gl::IntInstruction plain = listed;
    plain.markedGoal = values.encode("(sum[a,marker])");
    plain.pathVariables = 0;
    auto& integration = ((block).overallHashMemory).admissionMapIntegration;
    gl::IntegrationEntryMap entries = gl::admissionIntegrationRecordsAt(integration, pk, values);
    gl::payloadAt(entries, listed, values);
    gl::payloadAt(entries, plain, values);
    integration.assignRun(pk, gl::flattenIntegrationEntryMap(entries));
    std::filesystem::create_directories(".debug");
    gl::hashburst_dump::dumpExit(block);
    std::ifstream input(".debug/hashburst_trace.txt");
    ASSERT_TRUE(input.is_open());
    const std::string dump((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_TRUE(dump.find("] markedGoal=(fold[a,marker]) | data.size=1 | payload=0 | path=[R,rec]\n")
        != std::string::npos);
    ASSERT_TRUE(dump.find("] markedGoal=(sum[a,marker]) | data.size=1 | payload=0\n") != std::string::npos);
}

// A copy bound to one premise prints its two forbidden sets as ascending
// members; an unsplit instruction prints neither
// (D-579).
TEST(hashburst_dump, integration_instruction_line_carries_its_forbidden_sets) {
    gl::tests::IsolatedTestDirectory directory("hashburst_dump_forbidden_test");
    gl::Memory root;
    gl::Memory block;
    ((block).parentMemory) = &root;
    block.setExprKey("(fixture[u_1])");
    gl::ValueInterner& values = ((block).valueInterner);
    const int64_t pk = gl::mintTemplateKey(((block).templateInterner), ((block).nameMap),
        "(in2[u_a,marker,3])", "main");
    gl::IntInstruction split;
    gl::IntLogicalEntity entity;
    entity.category = values.encode("atomic");
    entity.signature = values.encode("(in2[u_a,marker,3])");
    entity.elements = { values.encode("u_a"), values.encode("marker"), values.encode("3") };
    entity.arity = 3;
    entity.definedSet = values.encode("N");
    split.data.push_back(entity);
    split.markedGoal = values.encode("(fold[a,marker])");
    split.forbidden.variables = (uint64_t{1} << 12) | (uint64_t{1} << 40);
    split.forbidden.levels = uint64_t{1} << 2;
    gl::IntInstruction plain = split;
    plain.markedGoal = values.encode("(sum[a,marker])");
    plain.forbidden = gl::IntegrationForbiddenSets{};
    auto& integration = ((block).overallHashMemory).admissionMapIntegration;
    gl::IntegrationEntryMap entries = gl::admissionIntegrationRecordsAt(integration, pk, values);
    gl::payloadAt(entries, split, values);
    gl::payloadAt(entries, plain, values);
    integration.assignRun(pk, gl::flattenIntegrationEntryMap(entries));
    std::filesystem::create_directories(".debug");
    gl::hashburst_dump::dumpExit(block);
    std::ifstream input(".debug/hashburst_trace.txt");
    ASSERT_TRUE(input.is_open());
    const std::string dump((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    ASSERT_TRUE(dump.find("] markedGoal=(fold[a,marker]) | data.size=1 | payload=0"
        " | forbiddenVariables=[12,40] | forbiddenLevels=[2]\n") != std::string::npos);
    ASSERT_TRUE(dump.find("] markedGoal=(sum[a,marker]) | data.size=1 | payload=0\n") != std::string::npos);
}
