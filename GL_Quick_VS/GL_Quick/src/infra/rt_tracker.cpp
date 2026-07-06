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

#include "infra/rt_tracker.hpp"

#include "memory.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace gl {
namespace rt_tracker {

int g_currentHashburstIndex = -1;
thread_local RTTracker* g_currentThreadTracker = nullptr;

namespace {

constexpr const char* RT_FOLDER          = ".rt";
constexpr const char* RT_FILE_EXTENSION  = ".log";
constexpr const char* RT_FILE_TMP_EXTENSION = ".log.tmp";

// Conservative cap for the filename component (folder + extension counted
// separately). Windows MAX_PATH is 260; we keep ~120 chars for the chain
// portion so the absolute path stays well under, even from a long worktree
// root like C:\Users\nikol\pycharmprojects\GL\.worktree\<branch>\.
constexpr std::size_t MAX_CHAIN_FILENAME_LEN = 120;

/// @brief Replace filesystem-unsafe characters with `_`.
///
/// The substitution set matches the union of Windows-reserved characters
/// (`<`, `>`, `:`, `"`, `/`, `\\`, `|`, `?`, `*`) plus the MPL-bracket
/// characters (`(`, `)`, `[`, `]`, `,`) that filesystems typically
/// accept but make the filename unergonomic to type / glob. Whitespace
/// is also replaced.
///
/// @param in The raw chain string.
/// @return A string where every member of the substitution set has been
///         replaced by `_`.
std::string scrubFilesystemUnsafe(const std::string& in) {
    static const char unsafe[] = "/\\:*?\"<>|()[], \t\n\r";
    std::string out;
    out.reserve(in.size());
    for (char c : in) {
        bool replace = false;
        for (const char* p = unsafe; *p; ++p) {
            if (c == *p) { replace = true; break; }
        }
        out.push_back(replace ? '_' : c);
    }
    return out;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// RTScope
// ---------------------------------------------------------------------------

RTScope::RTScope(RTTracker& tracker, const char* label)
    : tracker_(&tracker),
      section_index_(tracker.openSection_(label)) {
}

RTScope::~RTScope() {
    tracker_->closeSection_(section_index_);
}

// ---------------------------------------------------------------------------
// RTScopeHere
// ---------------------------------------------------------------------------

RTScopeHere::RTScopeHere(const char* label)
    : tracker_(g_currentThreadTracker),
      section_index_(-1) {
    if (tracker_) {
        section_index_ = tracker_->openSection_(label);
    }
}

RTScopeHere::~RTScopeHere() {
    if (tracker_) {
        tracker_->closeSection_(section_index_);
    }
}

// ---------------------------------------------------------------------------
// RTTracker
// ---------------------------------------------------------------------------

RTTracker::RTTracker(const Memory& body)
    : RTTracker(body,
                RTMeasurementParameters::RT_TIME_TRIGGER_SECONDS,
                RTMeasurementParameters::RT_MIN_PERCENTAGE) {
}

RTTracker::RTTracker(const Memory& body, int triggerSeconds, int minPercentage)
    : section_count_(0),
      open_depth_(0),
      t_start_(Clock::now()),
      t_last_event_(t_start_),
      t_last_refresh_(t_start_),
      chain_human_(buildChainHuman_(body)),
      chain_filename_(sanitizeForFilename_(chain_human_)),
      trigger_seconds_(triggerSeconds),
      min_percentage_(minPercentage),
      hashburst_index_(g_currentHashburstIndex),
      ever_dumped_(false) {
    // Publish ourselves as this thread's active tracker so RT_SCOPE_HERE
    // and RT_REFRESH_HERE in inner functions can reach us.
    g_currentThreadTracker = this;
}

RTTracker::~RTTracker() {
    // If the trigger fired during this call, emit one final snapshot
    // labelled `(finished)` so the on-disk file reflects the true
    // end-of-call state instead of a stale `(still running)` line from
    // the last in-flight refresh. Trackers whose call returned under
    // the trigger leave no file at all (ever_dumped_ stays false).
    //
    // Note: any still-open RTScope at this point is the caller's bug
    // (RAII scope close should have fired first); we tolerate it
    // silently rather than asserting so an upstream assert / exception
    // unwind isn't masked by a destructor-side failure.
    if (ever_dumped_) {
        writeSnapshot_(/*finished=*/true);
    }
    // Drop the thread-local pointer so subsequent RT_SCOPE_HERE /
    // RT_REFRESH_HERE calls on this thread are no-ops until the next
    // RTTracker is constructed.
    if (g_currentThreadTracker == this) {
        g_currentThreadTracker = nullptr;
    }
}

int RTTracker::openSection_(const char* label) {
    // Charge the wall-clock delta since the last event to whatever
    // scope is currently on top (if any) before pushing the new
    // section. This keeps exclusive self-time correct across nested
    // open/close transitions.
    chargeElapsedToTop_();

    // Find or allocate a row for this label. Label pointers are
    // expected to be string literals from the RT_SCOPE macro, so
    // pointer equality is the right comparison (no strcmp).
    int index = -1;
    for (int i = 0; i < section_count_; ++i) {
        if (sections_[i].label == label) {
            index = i;
            break;
        }
    }
    if (index < 0) {
        assert(section_count_ < RTMeasurementParameters::RT_MAX_SECTIONS
               && "rt_tracker: RT_MAX_SECTIONS exhausted; raise the cap or "
                  "reduce the number of distinct RT_SCOPE labels in this function");
        index = section_count_++;
        sections_[index] = {label, 0, 0, 0};
    }
    ++sections_[index].hits;

    assert(open_depth_ < RTMeasurementParameters::RT_MAX_SECTIONS);
    open_stack_[open_depth_++] = index;
    return index;
}

void RTTracker::closeSection_(int sectionIndex) {
    assert(open_depth_ > 0 && "rt_tracker: closeSection_ called with empty stack");
    assert(open_stack_[open_depth_ - 1] == sectionIndex
           && "rt_tracker: scope close order mismatch (RTScope objects "
              "destroyed out of construction order)");

    chargeElapsedToTop_();
    --open_depth_;

    refreshIfTriggered();
}

void RTTracker::chargeElapsedToTop_() {
    const auto now = Clock::now();
    if (open_depth_ > 0) {
        const auto dt = std::chrono::duration_cast<std::chrono::nanoseconds>(
                            now - t_last_event_)
                            .count();
        sections_[open_stack_[open_depth_ - 1]].self_ns += dt;
    }
    t_last_event_ = now;
}

void RTTracker::noteIterations(int n) {
    assert(open_depth_ > 0
           && "rt_tracker: noteIterations called with no open scope");
    sections_[open_stack_[open_depth_ - 1]].iterations += n;
}

void RTTracker::refreshIfTriggered() {
    const auto now = Clock::now();
    const auto elapsed_s = std::chrono::duration_cast<std::chrono::seconds>(
                               now - t_start_)
                               .count();
    if (elapsed_s < static_cast<long long>(trigger_seconds_)) return;

    // Throttle: at most one file rewrite per second, regardless of how
    // many RT_REFRESH_HERE calls fire from inner loops. The chrono::now()
    // + integer compare is cheap; the actual write (open + 1KB + flush +
    // rename) is what would dominate if every stack-pop dumped. The
    // throttle keeps the on-disk file fresh for a watching human while
    // adding negligible overhead per call site.
    const auto since_last_ms =
        std::chrono::duration_cast<std::chrono::milliseconds>(
            now - t_last_refresh_)
            .count();
    // The throttle limits SUBSEQUENT writes to at most one per second; the
    // first write (ever_dumped_ == false) must always land, otherwise a
    // tracker whose triggered lifetime is under a second never produces a
    // file. t_last_refresh_ is seeded to t_start_, so without this guard the
    // initial refresh is spuriously throttled.
    if (ever_dumped_ && since_last_ms < 1000) return;
    t_last_refresh_ = now;

    writeSnapshot_(/*finished=*/false);
}

void RTTracker::writeSnapshot_(bool finished) {
    ever_dumped_ = true;
    // Charge the time since the last event so the snapshot reflects
    // the latest state of the currently-open scope, not the state at
    // the previous scope boundary.
    chargeElapsedToTop_();

    const auto now = Clock::now();
    const double elapsed_seconds =
        std::chrono::duration<double>(now - t_start_).count();

    // The snapshot file path. `.rt/` may not yet exist (the main()
    // cleanup hook creates it; unit tests do not run that hook).
    std::error_code ec;
    std::filesystem::create_directories(RT_FOLDER, ec);
    if (ec) {
        // Instrumentation must never crash the prover. If .rt/ cannot
        // be created (permission, path-too-long, transient FS error),
        // skip this snapshot and hope the next refresh succeeds.
        return;
    }

    const std::string live_path =
        std::string(RT_FOLDER) + "/" + chain_filename_ + RT_FILE_EXTENSION;
    const std::string tmp_path =
        std::string(RT_FOLDER) + "/" + chain_filename_ + RT_FILE_TMP_EXTENSION;

    // Sort sections by descending self-time. We index by sort order so
    // the original sections_ array stays in insert order (helpful for
    // the human-readable comparison across refreshes).
    std::vector<int> order;
    order.reserve(static_cast<std::size_t>(section_count_));
    for (int i = 0; i < section_count_; ++i) order.push_back(i);
    std::sort(order.begin(), order.end(),
              [this](int a, int b) {
                  return sections_[a].self_ns > sections_[b].self_ns;
              });

    // Render. We use a stringstream so we can prepare the full body
    // before opening the temp file; that keeps the write window short.
    std::ostringstream ss;
    ss << "LB chain (root -> leaf):\n";
    {
        // Pretty-print the chain by splitting on the "__" separator
        // used in buildChainHuman_; the first element is "(root)".
        std::string remaining = chain_human_;
        const std::string sep = " -> ";
        std::size_t start = 0;
        while (start < remaining.size()) {
            const std::size_t pos = remaining.find(sep, start);
            const std::string token =
                (pos == std::string::npos)
                    ? remaining.substr(start)
                    : remaining.substr(start, pos - start);
            ss << "  " << token << "\n";
            if (pos == std::string::npos) break;
            start = pos + sep.size();
        }
    }
    ss << "\n";

    {
        char hdr[384];
        std::snprintf(hdr, sizeof(hdr),
                      "Hashburst index            : %d\n"
                      "Total elapsed in this call : %.2f s   %s\n"
                      "Trigger threshold          : %d s\n"
                      "Display threshold          : %d %% of total\n",
                      hashburst_index_,
                      elapsed_seconds,
                      finished ? "(finished)" : "(still running)",
                      trigger_seconds_, min_percentage_);
        ss << hdr;
    }
    // Currently-active scope chain: innermost-last. For mid-burst
    // snapshots this shows which function is on the CPU right now;
    // for the finished snapshot it's empty (all scopes are closed).
    ss << "Currently in               : ";
    if (open_depth_ == 0) {
        ss << "(no active scope)\n\n";
    } else {
        for (int i = 0; i < open_depth_; ++i) {
            if (i > 0) ss << " > ";
            ss << sections_[open_stack_[i]].label;
        }
        ss << "\n\n";
    }

    ss << "Section                                            | active | seconds | %total | hits | iter\n";
    ss << "---------------------------------------------------+--------+---------+--------+------+-------\n";

    // Pre-compute the active-row set so each printed row can be
    // marked. A section is active when its index is anywhere on the
    // open_stack_.
    bool is_active[RTMeasurementParameters::RT_MAX_SECTIONS] = {false};
    for (int i = 0; i < open_depth_; ++i) {
        is_active[open_stack_[i]] = true;
    }

    int     hidden_count = 0;
    int64_t hidden_ns    = 0;
    for (int idx : order) {
        const Section& s = sections_[idx];
        const double sec = static_cast<double>(s.self_ns) / 1e9;
        const double pct =
            (elapsed_seconds > 0.0) ? 100.0 * sec / elapsed_seconds : 0.0;
        if (pct < static_cast<double>(min_percentage_)) {
            ++hidden_count;
            hidden_ns += s.self_ns;
            continue;
        }
        const char* active_marker = is_active[idx] ? " *    " : "      ";
        char row[256];
        if (s.iterations > 0) {
            std::snprintf(row, sizeof(row),
                          "%-50s |%s | %7.2f | %6.2f | %4d | %6d\n",
                          s.label, active_marker, sec, pct, s.hits, s.iterations);
        } else {
            std::snprintf(row, sizeof(row),
                          "%-50s |%s | %7.2f | %6.2f | %4d | %6s\n",
                          s.label, active_marker, sec, pct, s.hits, "-");
        }
        ss << row;
    }
    ss << "---------------------------------------------------+--------+---------+--------+------+-------\n";
    if (hidden_count > 0) {
        const double hidden_sec = static_cast<double>(hidden_ns) / 1e9;
        const double hidden_pct =
            (elapsed_seconds > 0.0) ? 100.0 * hidden_sec / elapsed_seconds : 0.0;
        char tail[256];
        std::snprintf(tail, sizeof(tail),
                      "(%d sections each < %d %% of total; "
                      "%.2f s / %.2f %% combined)\n",
                      hidden_count, min_percentage_, hidden_sec, hidden_pct);
        ss << tail;
    }

    // Atomic write: write tmp, rename over live. Both the open and the
    // rename can transiently fail on Windows when an external process
    // (most commonly Windows Defender) is briefly holding the file open
    // for scanning between our writes. Retry the whole sequence a few
    // times before giving up — and on giving up, return silently rather
    // than asserting. This is instrumentation, not main-pipeline
    // correctness: a missed snapshot is recoverable (the next scope
    // close calls writeSnapshot_ again) and must never crash the prover.
    //
    // Rule-19 note. Both branches of the loop are part of the
    // contract: "snapshot written this attempt" vs "deferred to the
    // next refresh". Failure to dump is a defined outcome of the
    // diagnostic layer, not a hidden correctness defect.
    const std::string body_str = ss.str();
    constexpr int kMaxAttempts = 10;
    constexpr int kRetryDelayMs = 50;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        {
            std::ofstream out(tmp_path, std::ios::out | std::ios::trunc);
            if (!out.good()) {
                std::this_thread::sleep_for(
                    std::chrono::milliseconds(kRetryDelayMs));
                continue;
            }
            out << body_str;
            out.flush();
        }
        std::error_code rename_ec;
        std::filesystem::rename(tmp_path, live_path, rename_ec);
        if (!rename_ec) {
            // The next attribution window starts now.
            t_last_event_ = Clock::now();
            return;
        }
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kRetryDelayMs));
    }
    // All retries exhausted — leave any tmp behind as evidence and
    // return without crashing. The next refresh will try again.

    // The next attribution window starts now.
    t_last_event_ = Clock::now();
}

std::string RTTracker::buildChainHuman_(const Memory& body) {
    // Collect exprKey values walking parentMemory upward, then reverse
    // so the output is root -> leaf order. The root sentinel has an
    // empty exprKey + nullptr parent (per Rule 12); we render it as
    // the literal "(root)".
    std::vector<std::string> rev;
    const Memory* cur = &body;
    while (cur != nullptr) {
        if (cur->parentMemory == nullptr && cur->exprKey().empty()) {
            rev.push_back("(root)");
        } else {
            rev.push_back(cur->exprKey());
        }
        cur = cur->parentMemory;
    }
    std::reverse(rev.begin(), rev.end());

    std::string out;
    for (std::size_t i = 0; i < rev.size(); ++i) {
        if (i > 0) out += " -> ";
        out += rev[i];
    }
    return out;
}

std::string RTTracker::sanitizeForFilename_(const std::string& chain) {
    const std::string scrubbed = scrubFilesystemUnsafe(chain);
    if (scrubbed.size() <= MAX_CHAIN_FILENAME_LEN) return scrubbed;

    // Too long: keep a leading prefix that still names the outermost
    // LB (so `ls .rt/` groups by anchor), then append a short hash to
    // disambiguate.
    constexpr std::size_t HASH_HEX_LEN = 12;
    const std::size_t prefix_len = MAX_CHAIN_FILENAME_LEN - HASH_HEX_LEN - 2;

    const std::size_t hashed = std::hash<std::string>{}(chain);
    char hash_buf[32];
    std::snprintf(hash_buf, sizeof(hash_buf), "%012zx",
                  hashed & 0x0000FFFFFFFFFFFFull);

    std::string out;
    out.reserve(MAX_CHAIN_FILENAME_LEN);
    out.append(scrubbed.substr(0, prefix_len));
    out += "__";
    out += hash_buf;
    return out;
}

} // namespace rt_tracker
} // namespace gl
