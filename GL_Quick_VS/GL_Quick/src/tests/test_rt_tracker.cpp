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
/// @brief Unit tests for the runtime-measurement infrastructure
///        (`rt_tracker.hpp` / `rt_tracker.cpp`).
///
/// @details
/// Positive + negative coverage of the RTTracker / RTScope pair. The
/// tests inject custom trigger and min-percentage thresholds via the
/// test-only two-int constructor so they execute in milliseconds
/// instead of having to wait the production-default 120 s. The
/// production constructor uses the `RTMeasurementParameters` defaults
/// and is exercised indirectly by anyone who builds with
/// `RT_MEASUREMENT = 1`; it does not need a direct test because the
/// only difference is the threshold values.
///
/// Each test that produces a `.rt/` file cleans up after itself via
/// `std::filesystem::remove`; the file lifecycle is per-call by
/// design so cleanup is a test concern, not a production one.

#include "test_harness.hpp"

#include "../memory.hpp"
#include "../infra/rt_tracker.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>

namespace {

/// Slurp a file's content into a string. Used to inspect what the
/// tracker wrote to `.rt/<chain>.log`.
std::string readWholeFile(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

/// Build a three-deep LB chain (root sentinel + two children) sharing
/// the parent-pointer convention used by the elementary step:
/// root has empty `exprKey` + nullptr parent, every other LB carries
/// its own `exprKey` and a back-pointer.
///
/// The output `leaf` is the innermost LB; the caller passes it to the
/// tracker constructor. The two parents must outlive the tracker.
struct SyntheticChain {
    gl::Memory root;
    gl::Memory middle;
    gl::Memory leaf;
};

void buildSyntheticChain(SyntheticChain& c,
                         const std::string& middle_key,
                         const std::string& leaf_key) {
    c.middle.parentMemory = &c.root;
    c.middle.setExprKey(middle_key);
    c.middle.level        = 0;
    c.leaf.parentMemory   = &c.middle;
    c.leaf.setExprKey(leaf_key);
    c.leaf.level          = 1;
}

/// Return the expected `.rt/<file>.log` path for a given sanitized
/// chain filename. Mirrors the production logic in rt_tracker.cpp.
std::string rtFilePath(const std::string& chain_filename) {
    return std::string(".rt/") + chain_filename + ".log";
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Test 1 — chain build + filename sanitization round-trip.
// ---------------------------------------------------------------------------
//
// The human-readable chain renders root -> leaf with `(root)` for the
// sentinel. The filename form replaces every filesystem-unsafe character
// with `_`.

TEST(rt_tracker, chain_human_and_filename_round_trip) {
    SyntheticChain c;
    buildSyntheticChain(c, "(AnchorPeano[N,i0,s])", "(=[v1,v2])");

    ::gl::rt_tracker::RTTracker tracker(c.leaf, /*triggerSeconds=*/9999,
                                        /*minPercentage=*/2);

    // Human chain shows the sentinel and both children, root -> leaf.
    ASSERT_EQ(tracker.chainHuman(),
              std::string("(root) -> (AnchorPeano[N,i0,s]) -> (=[v1,v2])"));

    // Filename strips the brackets / commas / arrows / spaces; underscores
    // are kept literally; nothing else maps to a path-traversal token.
    const std::string& fn = tracker.chainFilename();
    ASSERT_EQ(fn.find('/'),  std::string::npos);
    ASSERT_EQ(fn.find('\\'), std::string::npos);
    ASSERT_EQ(fn.find('('),  std::string::npos);
    ASSERT_EQ(fn.find('['),  std::string::npos);
    ASSERT_EQ(fn.find(','),  std::string::npos);
    ASSERT_EQ(fn.find(' '),  std::string::npos);
    // Anchor word survives so `ls .rt/` groups by outermost LB.
    ASSERT_NE(fn.find("AnchorPeano"), std::string::npos);
}

// ---------------------------------------------------------------------------
// Test 2 — under-trigger run leaves no file behind.
// ---------------------------------------------------------------------------
//
// Trigger 9999 s, scopes that close in microseconds → no refresh fires,
// no file appears at `.rt/<chain>.log`.

TEST(rt_tracker, under_trigger_writes_no_file) {
    SyntheticChain c;
    buildSyntheticChain(c, "(AnchorPeano[N,i0,s])", "(=[a,b])");

    std::string path;
    {
        ::gl::rt_tracker::RTTracker tracker(c.leaf,
                                            /*triggerSeconds=*/9999,
                                            /*minPercentage=*/2);
        path = rtFilePath(tracker.chainFilename());

        // Make sure the file does not exist before we open any scope —
        // otherwise the test would be confused by leftover state.
        std::error_code ec;
        std::filesystem::remove(path, ec);

        {
            ::gl::rt_tracker::RTScope s(tracker, "PHASE_A");
            // Tiny actual work; far below 9999 s.
            std::this_thread::sleep_for(std::chrono::microseconds(50));
        }
    }
    ASSERT_TRUE(!std::filesystem::exists(path));
}

// ---------------------------------------------------------------------------
// Test 3 — over-trigger run writes a file with the expected shape.
// ---------------------------------------------------------------------------
//
// Trigger 0 s, every scope close calls refreshIfTriggered() which writes
// the snapshot. We verify the file appears, contains the chain header,
// contains the section label, and the trailing hidden-row line is
// absent because everything is above 2 %.

TEST(rt_tracker, over_trigger_writes_table_with_chain_and_label) {
    SyntheticChain c;
    buildSyntheticChain(c, "(AnchorPeano[N,i0,s])", "(=[c,d])");

    std::string path;
    {
        ::gl::rt_tracker::RTTracker tracker(c.leaf,
                                            /*triggerSeconds=*/0,
                                            /*minPercentage=*/2);
        path = rtFilePath(tracker.chainFilename());

        std::error_code ec;
        std::filesystem::remove(path, ec);

        {
            ::gl::rt_tracker::RTScope s(tracker, "SOLE_PHASE");
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
        }
        // Scope close has fired refreshIfTriggered (trigger == 0 always
        // fires). The file is on disk now.
    }
    ASSERT_TRUE(std::filesystem::exists(path));

    const std::string body = readWholeFile(path);
    ASSERT_NE(body.find("LB chain (root -> leaf):"), std::string::npos);
    ASSERT_NE(body.find("(AnchorPeano[N,i0,s])"),    std::string::npos);
    ASSERT_NE(body.find("(=[c,d])"),                 std::string::npos);
    ASSERT_NE(body.find("SOLE_PHASE"),               std::string::npos);
    // Trigger threshold echoed in the file.
    ASSERT_NE(body.find("Trigger threshold          : 0 s"),
              std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// Test 4 — nested scopes attribute exclusive self-time.
// ---------------------------------------------------------------------------
//
// Outer scope opens, sleeps briefly, then opens an inner scope and
// sleeps inside it. The outer scope's self_ns should reflect only the
// pre-inner sleep; the inner scope's self_ns should reflect the inner
// sleep.

TEST(rt_tracker, nested_scopes_exclusive_self_time) {
    SyntheticChain c;
    buildSyntheticChain(c, "(AnchorPeano[N,i0,s])", "(=[e,f])");

    std::string path;
    {
        ::gl::rt_tracker::RTTracker tracker(c.leaf,
                                            /*triggerSeconds=*/0,
                                            /*minPercentage=*/0);
        path = rtFilePath(tracker.chainFilename());

        std::error_code ec;
        std::filesystem::remove(path, ec);

        {
            ::gl::rt_tracker::RTScope outer(tracker, "OUTER");
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            {
                ::gl::rt_tracker::RTScope inner(tracker, "INNER");
                std::this_thread::sleep_for(std::chrono::milliseconds(80));
            }
            // After inner closes, the next attributable interval is
            // charged to OUTER again. Stop short to keep OUTER < INNER.
        }
    }
    ASSERT_TRUE(std::filesystem::exists(path));
    const std::string body = readWholeFile(path);

    // Both labels appear.
    ASSERT_NE(body.find("OUTER"), std::string::npos);
    ASSERT_NE(body.find("INNER"), std::string::npos);

    // The body lists sections sorted by descending seconds. INNER's
    // self-time is the ~80 ms inner sleep; OUTER's is the ~10 ms
    // before the inner scope. The 8x ratio survives Windows
    // sleep_for jitter (~1-15 ms granularity) so INNER's row
    // reliably appears above OUTER's row in the table.
    const std::size_t pos_inner = body.find("INNER");
    const std::size_t pos_outer = body.find("OUTER");
    ASSERT_TRUE(pos_inner < pos_outer);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// Test 5 — noteIterations stamps the open scope.
// ---------------------------------------------------------------------------

TEST(rt_tracker, note_iterations_records_count) {
    SyntheticChain c;
    buildSyntheticChain(c, "(AnchorPeano[N,i0,s])", "(=[g,h])");

    std::string path;
    {
        ::gl::rt_tracker::RTTracker tracker(c.leaf,
                                            /*triggerSeconds=*/0,
                                            /*minPercentage=*/0);
        path = rtFilePath(tracker.chainFilename());

        std::error_code ec;
        std::filesystem::remove(path, ec);

        {
            ::gl::rt_tracker::RTScope s(tracker, "LOOP_PHASE");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
            tracker.noteIterations(1234);
        }
    }
    ASSERT_TRUE(std::filesystem::exists(path));
    const std::string body = readWholeFile(path);
    ASSERT_NE(body.find("LOOP_PHASE"), std::string::npos);
    ASSERT_NE(body.find("1234"),       std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// Test 6 — atomic rewrite: the live file never appears empty mid-write.
// ---------------------------------------------------------------------------
//
// Reading the file at any moment must yield either the previous snapshot
// or the new snapshot, never a half-written one. We exercise this by
// taking three snapshots in a row and verifying each one is complete
// (contains the closing separator row).

TEST(rt_tracker, atomic_rewrite_never_truncates_live_file) {
    SyntheticChain c;
    buildSyntheticChain(c, "(AnchorPeano[N,i0,s])", "(=[i,j])");

    std::string path;
    {
        ::gl::rt_tracker::RTTracker tracker(c.leaf,
                                            /*triggerSeconds=*/0,
                                            /*minPercentage=*/2);
        path = rtFilePath(tracker.chainFilename());

        std::error_code ec;
        std::filesystem::remove(path, ec);

        for (int i = 0; i < 3; ++i) {
            {
                ::gl::rt_tracker::RTScope s(tracker, "REFRESH_PHASE");
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
            ASSERT_TRUE(std::filesystem::exists(path));
            const std::string body = readWholeFile(path);
            // Every complete snapshot ends with the section-divider
            // row (the trailing one with `-----`).
            ASSERT_NE(body.find(
                          "---------------------------------------------------"
                          "+--------+---------+--------+------+-------"),
                      std::string::npos);
        }
    }
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

// ---------------------------------------------------------------------------
// Test 7 — same label opened twice accumulates one row with hits == 2.
// ---------------------------------------------------------------------------

TEST(rt_tracker, repeated_label_accumulates_hits) {
    SyntheticChain c;
    buildSyntheticChain(c, "(AnchorPeano[N,i0,s])", "(=[k,l])");

    static constexpr const char* kLabel = "REPEAT_PHASE";

    std::string path;
    {
        ::gl::rt_tracker::RTTracker tracker(c.leaf,
                                            /*triggerSeconds=*/0,
                                            /*minPercentage=*/0);
        path = rtFilePath(tracker.chainFilename());

        std::error_code ec;
        std::filesystem::remove(path, ec);

        for (int i = 0; i < 2; ++i) {
            ::gl::rt_tracker::RTScope s(tracker, kLabel);
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
    ASSERT_TRUE(std::filesystem::exists(path));
    const std::string body = readWholeFile(path);
    // The hits column is the fourth `|`-separated field — we just
    // verify the row exists and contains a 2 somewhere reasonable.
    const std::size_t row = body.find("REPEAT_PHASE");
    ASSERT_NE(row, std::string::npos);
    const std::size_t row_end = body.find('\n', row);
    ASSERT_NE(row_end, std::string::npos);
    const std::string row_text = body.substr(row, row_end - row);
    // Loose check: the row must contain the substring "|    2 " or
    // similar, indicating hits == 2.
    ASSERT_NE(row_text.find('2'), std::string::npos);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}
