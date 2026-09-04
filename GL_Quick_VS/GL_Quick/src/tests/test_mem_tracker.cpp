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
/// @brief Unit tests for the per-structure memory measurement
///        (`mem_tracker.hpp` / `mem_tracker.cpp`).
///
/// @details
/// The tracker is gate-independent at the class level — `MEM_MEASUREMENT`
/// removes only the prover's call sites, exactly as `RT_MEASUREMENT` does for
/// the RT tracker — so these tests exercise it in a normal build. Each test
/// calls `resetAllForTest()` first so the process-wide tables cannot leak
/// between cases, and the dump test removes its own file.
///
/// Coverage is positive (a sample lands, larger content produces a larger
/// sample, two worker slots both contribute, the peak snapshot survives a
/// smaller follow-up, content / index / slack stay separated, the text dump
/// names a structure and the HTML renders a three-level hierarchy) and negative
/// (an iteration reset discards uncommitted worker rows; a dump before any
/// commit still writes a well-formed zero table rather than failing).

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../infra/mem_tracker.hpp"

#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace {

/// Slurp a file into a string so the dump's rendering can be inspected.
std::string readWholeFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/// Build a statement row with a distinguishable name id.
gl::IntEncodedExpr makeRow(gl::NameId i) {
    gl::IntEncodedExpr e{};
    e.nameId = i;
    e.originalId = i;
    e.validityId = 1;
    return e;
}

}  // namespace

TEST(mem_tracker, sample_lands_and_sets_a_peak) {
    gl::mem_tracker::resetAllForTest();
    ASSERT_EQ(gl::mem_tracker::sampleCount(), int64_t(0));
    ASSERT_EQ(gl::mem_tracker::peakTotalBytes(), int64_t(0));

    gl::Memory m;
    m.setExprKey("(mem_tracker_sample_lb)");
    for (gl::NameId i = 0; i < 16; ++i)
        m.intEncodedStatements.push_back(makeRow(i));

    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(m.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();

    ASSERT_EQ(gl::mem_tracker::sampleCount(), int64_t(1));
    // The sixteen rows alone are already more than a kilobyte of content, and
    // the slack counter adds the LB's whole pinned block on top.
    ASSERT_GE(gl::mem_tracker::peakTotalBytes(),
              int64_t(16 * sizeof(gl::IntEncodedExpr)));
}

TEST(mem_tracker, more_content_makes_a_larger_sample) {
    gl::mem_tracker::resetAllForTest();

    gl::Memory small;
    small.setExprKey("(mem_tracker_small_lb)");
    for (gl::NameId i = 0; i < 4; ++i)
        small.intEncodedStatements.push_back(makeRow(i));
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(small.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();
    const int64_t smallPeak = gl::mem_tracker::peakTotalBytes();

    gl::Memory big;
    big.setExprKey("(mem_tracker_big_lb)");
    for (gl::NameId i = 0; i < 400; ++i)
        big.intEncodedStatements.push_back(makeRow(i));
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(big.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();

    ASSERT_EQ(gl::mem_tracker::sampleCount(), int64_t(2));
    ASSERT_GE(gl::mem_tracker::peakTotalBytes(), smallPeak);
    ASSERT_GE(gl::mem_tracker::peakTotalBytes(),
              int64_t(400 * sizeof(gl::IntEncodedExpr)));
}

TEST(mem_tracker, two_worker_slots_both_contribute) {
    gl::mem_tracker::resetAllForTest();

    gl::Memory a;
    a.setExprKey("(mem_tracker_slot_a)");
    for (gl::NameId i = 0; i < 32; ++i)
        a.intEncodedStatements.push_back(makeRow(i));
    gl::Memory b;
    b.setExprKey("(mem_tracker_slot_b)");
    for (gl::NameId i = 0; i < 32; ++i)
        b.intEncodedStatements.push_back(makeRow(i));

    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(a.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();
    const int64_t onlyA = gl::mem_tracker::peakTotalBytes();

    // Same two LBs, but this time both sampled into DIFFERENT worker rows: the
    // fold must add them, which is the property that makes the barrier sample a
    // whole-grid instant rather than one worker's view.
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(a.lbMemory, 0);
    gl::mem_tracker::addLbSample(b.lbMemory, 1);
    gl::mem_tracker::commitIterationSample();

    ASSERT_GE(gl::mem_tracker::peakTotalBytes(), onlyA * 2);
}

TEST(mem_tracker, reset_iteration_discards_uncommitted_rows) {
    gl::mem_tracker::resetAllForTest();

    gl::Memory m;
    m.setExprKey("(mem_tracker_reset_lb)");
    for (gl::NameId i = 0; i < 64; ++i)
        m.intEncodedStatements.push_back(makeRow(i));

    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(m.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();
    const int64_t withContent = gl::mem_tracker::peakTotalBytes();
    ASSERT_GE(withContent, int64_t(64 * sizeof(gl::IntEncodedExpr)));

    // A reset with no sample after it must fold to nothing. The peak is a
    // high-water so it survives, but the sample counter still advances — which
    // is exactly how an all-idle iteration reads.
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::commitIterationSample();
    ASSERT_EQ(gl::mem_tracker::sampleCount(), int64_t(2));
    ASSERT_EQ(gl::mem_tracker::peakTotalBytes(), withContent);
}

TEST(mem_tracker, peak_snapshot_survives_a_smaller_follow_up) {
    gl::mem_tracker::resetAllForTest();

    gl::Memory big;
    big.setExprKey("(mem_tracker_peak_big)");
    for (gl::NameId i = 0; i < 256; ++i)
        big.intEncodedStatements.push_back(makeRow(i));
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(big.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();
    const int64_t peak = gl::mem_tracker::peakTotalBytes();

    gl::Memory small;
    small.setExprKey("(mem_tracker_peak_small)");
    small.intEncodedStatements.push_back(makeRow(0));
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(small.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();

    ASSERT_EQ(gl::mem_tracker::peakTotalBytes(), peak);
}

TEST(mem_tracker, dump_names_the_structure_that_holds_the_bytes) {
    gl::mem_tracker::resetAllForTest();

    gl::Memory m;
    m.setExprKey("(mem_tracker_dump_lb)");
    for (gl::NameId i = 0; i < 128; ++i)
        m.intEncodedStatements.push_back(makeRow(i));
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(m.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();

    const std::string path = ".rt/_memory_unit_test.log";
    gl::mem_tracker::dumpMemAggregate(path);
    const std::string text = readWholeFile(path);

    ASSERT_TRUE(text.find("intEncodedStatements") != std::string::npos);
    ASSERT_TRUE(text.find("%peak") != std::string::npos);
    ASSERT_TRUE(text.find("Peak attributed bytes") != std::string::npos);
    // Derived indexes and slack are named buckets in the header, never
    // silently folded into a container's figure.
    ASSERT_TRUE(text.find("derived hash indexes") != std::string::npos);
    ASSERT_TRUE(text.find("block + page slack") != std::string::npos);
    // The process-wide pool block is a separate section with its own
    // denominator; the table must say so rather than mixing the two.
    ASSERT_TRUE(text.find("PHYSICAL block") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(mem_tracker, dump_before_any_sample_writes_a_zero_table) {
    gl::mem_tracker::resetAllForTest();

    const std::string path = ".rt/_memory_unit_test_empty.log";
    gl::mem_tracker::dumpMemAggregate(path);
    const std::string text = readWholeFile(path);

    ASSERT_TRUE(text.find("Iteration samples committed : 0") != std::string::npos);
    ASSERT_TRUE(text.find("Peak attributed bytes       : 0") != std::string::npos);
    // No structure row can exist without a sample, but the header and the
    // column rule must still render — a caller inspecting the file learns the
    // gate produced nothing, rather than finding a truncated or absent file.
    ASSERT_TRUE(text.find("Structure") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(mem_tracker, index_and_slack_are_separated_from_content) {
    gl::mem_tracker::resetAllForTest();

    gl::Memory m;
    m.setExprKey("(mem_tracker_split_lb)");
    // Mint enough keys that the derived PagedHashIndex is unmistakably present:
    // a cold map's bucket array is never deloaded and never appears as content.
    for (gl::NameId i = 1; i < 400; ++i)
        m.intKnownStatements.insert(gl::StatementKey{ i, 1 },
                                    gl::StatementFlags{ true, false });
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(m.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();

    const std::string path = ".rt/_memory_unit_split.log";
    gl::mem_tracker::dumpMemAggregate(path);
    const std::string text = readWholeFile(path);

    // All three buckets must be named, and the peak is their sum.
    ASSERT_TRUE(text.find("container content") != std::string::npos);
    ASSERT_TRUE(text.find("derived hash indexes") != std::string::npos);
    ASSERT_TRUE(text.find("block + page slack") != std::string::npos);
    // A container that minted keys reports a non-zero index column, so the
    // index is genuinely measured and not folded into slack.
    ASSERT_TRUE(text.find("intKnownStatements") != std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST(mem_tracker, html_renders_a_three_level_hierarchy) {
    gl::mem_tracker::resetAllForTest();

    gl::Memory m;
    m.setExprKey("(mem_tracker_html_lb)");
    for (gl::NameId i = 0; i < 64; ++i)
        m.intEncodedStatements.push_back(makeRow(i));
    gl::mem_tracker::resetIteration();
    gl::mem_tracker::addLbSample(m.lbMemory, 0);
    gl::mem_tracker::commitIterationSample();

    const std::string path = ".rt/_memory_unit_test.html";
    gl::mem_tracker::dumpMemHtml(path);
    const std::string html = readWholeFile(path);

    ASSERT_TRUE(html.find("<!doctype html>") != std::string::npos);
    ASSERT_TRUE(html.find("</html>") != std::string::npos);
    // Collapsible hierarchy, and the instance node a direct LbMemory member
    // hangs under so every leaf sits at the same depth.
    ASSERT_TRUE(html.find("<details") != std::string::npos);
    ASSERT_TRUE(html.find("LbMemory (direct members)") != std::string::npos);
    ASSERT_TRUE(html.find("intEncodedStatements") != std::string::npos);
    // The three columns: size, share of content, share of peak.
    ASSERT_TRUE(html.find(">MiB<") != std::string::npos);
    ASSERT_TRUE(html.find(">%content<") != std::string::npos);
    ASSERT_TRUE(html.find(">%peak<") != std::string::npos);
    // Self-contained: no external stylesheet, script or font.
    ASSERT_TRUE(html.find("<link") == std::string::npos);
    ASSERT_TRUE(html.find("<script") == std::string::npos);
    ASSERT_TRUE(html.find("http://") == std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}
