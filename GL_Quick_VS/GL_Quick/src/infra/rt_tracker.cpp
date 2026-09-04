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
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_map>
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

// ---------------------------------------------------------------------------
// Process-wide RT aggregate — the cross-burst sink (see resetRtAggregate /
// dumpRtAggregate in the header). Keyed by (label string, parent label
// string) because identical literals in different translation units may
// carry different pointers; the strcmp cost sits on the cold per-burst
// destructor path only. All state under one mutex — no atomics needed.
// ---------------------------------------------------------------------------

struct AggRow {
    /// Full open-scope path root → leaf, labels joined with " > "
    /// (e.g. "REQGEN_BATCH3_LOCAL_X_MAIL > GENERATE_ENCODED_REQUESTS_STATIC
    /// > STATIC_REQGEN_PAIRING_MERGE"). Path-keying — not direct-parent
    /// keying — so a shared interior label stays attributed to the batch
    /// that owns the whole chain.
    std::string path;
    int64_t     self_ns;
    int64_t     hits;
    int64_t     iterations;
};

struct AggState {
    std::mutex          mtx;
    std::vector<AggRow> rows;
    /// path -> index into `rows` (the fold is per burst per section; a
    /// linear path scan over thousands of rows would dominate the fold).
    std::unordered_map<std::string, int> index;
    int64_t             tracker_calls   = 0;
    int64_t             tracker_life_ns = 0;
    /// Deepest scope nesting any tracker reached (virtual frames included).
    int                 peak_open_depth = 0;
    /// Single-timeline wall-clock accumulated per phase sweep (proveKernel
    /// registers each iteration's barrier-to-barrier seconds).
    double              phase1_wall_s   = 0.0;
    double              phase3_wall_s   = 0.0;
    double              phase2_wall_s   = 0.0;
};

/// Which phase tree a full aggregate path belongs to, by its ROOT label:
/// 1 = the phase-1 sweep's trackers, 3 = phase 3's, 2 = everything else
/// (the performElem2 hashburst tracker's REQGEN rows on the CPU route).
int phaseOfAggPath(const std::string& path) {
    if (path.rfind("PRE_FIXPOINT_MAIL_ABSORB", 0) == 0
        || path.rfind("PH1_", 0) == 0)
        return 1;
    if (path.rfind("POST_FIXPOINT_MAIL_FLUSH", 0) == 0
        || path.rfind("REACT_TO_HYPO", 0) == 0
        || path.rfind("END_OF_BURST_SANITIZE", 0) == 0
        || path.rfind("PH3_", 0) == 0)
        return 3;
    return 2;
}

/// Meyers singleton so unit tests and production share one instance
/// without a static-init-order hazard.
AggState& aggState() {
    static AggState s;
    return s;
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
    : sections_(new Section[RTMeasurementParameters::RT_MAX_SECTIONS]),
      section_count_(0),
      row_cache_(new int[kRowCacheSize]),
      open_depth_(0),
      virtual_depth_(0),
      peak_open_depth_(0),
      t_start_(Clock::now()),
      t_last_event_(t_start_),
      t_last_refresh_(t_start_),
      chain_human_(buildChainHuman_(body)),
      chain_filename_(sanitizeForFilename_(chain_human_)),
      trigger_seconds_(triggerSeconds),
      min_percentage_(minPercentage),
      hashburst_index_(g_currentHashburstIndex),
      ever_dumped_(false) {
    for (int i = 0; i < kRowCacheSize; ++i) row_cache_[i] = -1;
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
    // Fold this call's sections into the process-wide aggregate —
    // for EVERY tracker, trigger-independent, so the aggregate covers
    // all bursts. Charge the tail first so the last open window (a
    // tracker that never dumped has never charged) is attributed.
    chargeElapsedToTop_();
    accumulateAggregate_();
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

    // Depth saturation: a re-entrant production chain (the deposit door
    // through class-update products) can nest deeper than the stack cap.
    // Instrumentation must never crash a run the production build
    // completes, so beyond the cap a scope is VIRTUAL — not pushed, its
    // time folds into the innermost tracked section — and the true peak
    // depth is recorded as the evidence of how deep the chain went.
    if (open_depth_ >= RTMeasurementParameters::RT_MAX_OPEN_DEPTH) {
        ++virtual_depth_;
        if (open_depth_ + virtual_depth_ > peak_open_depth_)
            peak_open_depth_ = open_depth_ + virtual_depth_;
        return -1;
    }

    // Find or allocate a row for this (label, parent) pair. Label
    // pointers are expected to be string literals from the RT_SCOPE
    // macro, so pointer equality is the right comparison (no strcmp).
    // Keying by parent as well keeps one label opened under two
    // different enclosing scopes (e.g. an interior request-generation
    // scope under two REQGEN batches) as two separately-attributed rows.
    // Recursion collapsing: a label already open on the stack re-uses its
    // live section instead of minting a new (label, parent) row. Without
    // this, a re-entrant call chain (the deposit door re-entering itself
    // through class-update products) mints a fresh row set per recursion
    // level and exhausts RT_MAX_SECTIONS; with it, self-time folds into
    // the first occurrence's row — standard flat-recursion profiler
    // semantics. The close-order validation still holds because the same
    // index is pushed again.
    for (int d = open_depth_ - 1; d >= 0; --d) {
        if (sections_[open_stack_[d]].label == label) {
            ++sections_[open_stack_[d]].hits;
            open_stack_[open_depth_++] = open_stack_[d];
            if (open_depth_ > peak_open_depth_) peak_open_depth_ = open_depth_;
            return open_stack_[open_depth_ - 1];
        }
    }

    const int parentIdx = (open_depth_ > 0) ? open_stack_[open_depth_ - 1] : -1;
    // (label pointer, parent) -> row: direct-mapped cache first, the
    // linear scan only on a miss (the scan is O(rows) and a tracker with
    // thousands of rows would otherwise pay it on every scope open).
    const std::uintptr_t lp = reinterpret_cast<std::uintptr_t>(label);
    const int slot = static_cast<int>(
        ((lp >> 4) ^ (lp >> 20) ^ (static_cast<std::uintptr_t>(parentIdx + 1) * 0x9E3779B1u))
        & static_cast<std::uintptr_t>(kRowCacheSize - 1));
    int index = -1;
    {
        const int cached = row_cache_[slot];
        if (cached >= 0 && sections_[cached].label == label
            && sections_[cached].parentIdx == parentIdx) {
            index = cached;
        }
    }
    if (index < 0) {
        for (int i = 0; i < section_count_; ++i) {
            if (sections_[i].label == label && sections_[i].parentIdx == parentIdx) {
                index = i;
                break;
            }
        }
    }
    if (index < 0) {
        assert(section_count_ < RTMeasurementParameters::RT_MAX_SECTIONS
               && "rt_tracker: RT_MAX_SECTIONS exhausted; raise the cap or "
                  "reduce the number of distinct RT_SCOPE labels in this function");
        index = section_count_++;
        sections_[index] = {label, 0, 0, 0, parentIdx};
    }
    row_cache_[slot] = index;
    ++sections_[index].hits;

    open_stack_[open_depth_++] = index;
    if (open_depth_ > peak_open_depth_) peak_open_depth_ = open_depth_;
    return index;
}

void RTTracker::closeSection_(int sectionIndex) {
    // Virtual (depth-saturated) scope: nothing was pushed; unwind the
    // virtual counter and leave the tracked stack untouched.
    if (sectionIndex < 0) {
        assert(virtual_depth_ > 0
               && "rt_tracker: virtual-scope close without a virtual open");
        --virtual_depth_;
        return;
    }
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

void RTTracker::noteIterations(int64_t n) {
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
        // Parent-keyed rows render as "PARENT > LABEL" so one label
        // opened under two different enclosing scopes stays two
        // distinguishable rows in the table.
        char name[160];
        if (s.parentIdx >= 0) {
            std::snprintf(name, sizeof(name), "%s > %s",
                          sections_[s.parentIdx].label, s.label);
        } else {
            std::snprintf(name, sizeof(name), "%s", s.label);
        }
        char row[320];
        if (s.iterations > 0) {
            std::snprintf(row, sizeof(row),
                          "%-50s |%s | %7.2f | %6.2f | %4d | %6lld\n",
                          name, active_marker, sec, pct, s.hits,
                          static_cast<long long>(s.iterations));
        } else {
            std::snprintf(row, sizeof(row),
                          "%-50s |%s | %7.2f | %6.2f | %4d | %6s\n",
                          name, active_marker, sec, pct, s.hits, "-");
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

void RTTracker::accumulateAggregate_() {
    AggState& st = aggState();
    std::lock_guard<std::mutex> lock(st.mtx);
    st.tracker_calls += 1;
    st.tracker_life_ns += std::chrono::duration_cast<std::chrono::nanoseconds>(
        Clock::now() - t_start_).count();
    if (peak_open_depth_ > st.peak_open_depth)
        st.peak_open_depth = peak_open_depth_;
    for (int i = 0; i < section_count_; ++i) {
        const Section& s = sections_[i];
        // Build the full root -> leaf path by walking parentIdx. Depth is
        // bounded by RT_MAX_SECTIONS; the chain walk is cold (per burst).
        std::string path(s.label);
        for (int p = s.parentIdx; p >= 0; p = sections_[p].parentIdx) {
            path.insert(0, " > ");
            path.insert(0, sections_[p].label);
        }
        AggRow* row = nullptr;
        {
            const auto it = st.index.find(path);
            if (it != st.index.end()) row = &st.rows[static_cast<std::size_t>(it->second)];
        }
        if (row == nullptr) {
            st.index.emplace(path, static_cast<int>(st.rows.size()));
            st.rows.push_back(AggRow{ path, 0, 0, 0 });
            row = &st.rows.back();
        }
        row->self_ns += s.self_ns;
        row->hits += s.hits;
        row->iterations += s.iterations;
    }
}

void resetRtAggregate() {
    AggState& st = aggState();
    std::lock_guard<std::mutex> lock(st.mtx);
    st.rows.clear();
    st.index.clear();
    st.tracker_calls = 0;
    st.tracker_life_ns = 0;
    st.peak_open_depth = 0;
    st.phase1_wall_s = 0.0;
    st.phase2_wall_s = 0.0;
    st.phase3_wall_s = 0.0;
}

void addRtPhaseWallSeconds(int phase, double seconds) {
    AggState& st = aggState();
    std::lock_guard<std::mutex> lock(st.mtx);
    if (phase == 1) st.phase1_wall_s += seconds;
    else if (phase == 2) st.phase2_wall_s += seconds;
    else if (phase == 3) st.phase3_wall_s += seconds;
    else assert(false && "addRtPhaseWallSeconds: phase must be 1, 2 or 3");
}

void dumpRtAggregate(const std::string& path) {
    AggState& st = aggState();
    std::lock_guard<std::mutex> lock(st.mtx);

    std::vector<int> order;
    order.reserve(st.rows.size());
    for (int i = 0; i < static_cast<int>(st.rows.size()); ++i)
        order.push_back(i);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return st.rows[static_cast<std::size_t>(a)].self_ns
             > st.rows[static_cast<std::size_t>(b)].self_ns;
    });

    int64_t total_ns = 0;
    int64_t phase_ns[4] = { 0, 0, 0, 0 };
    for (const AggRow& r : st.rows) {
        total_ns += r.self_ns;
        phase_ns[phaseOfAggPath(r.path)] += r.self_ns;
    }

    std::ostringstream ss;
    ss << "RT aggregate (cross-burst, all trackers, trigger-independent)\n";
    ss << "\n";
    ss << "UNITS. 'seconds' = EXCLUSIVE self-time of that section, summed across\n";
    ss << "ALL worker threads and ALL bursts (worker-seconds). Several workers run\n";
    ss << "concurrently, so these sums EXCEED single-timeline wall-clock; within\n";
    ss << "one thread nothing overlaps and nothing is double-counted (a nested\n";
    ss << "row's time is excluded from its parent's row).\n";
    ss << "100% ('%attr' denominator) = 'Total attributed' below, in worker-seconds.\n";
    ss << "'%ph' = the row's share of ITS OWN phase tree (same-unit denominator:\n";
    ss << "the phase-1 / phase-3 / other attributed subtotal on its line below).\n";
    ss << "'wall~s' = the row's ESTIMATED single-timeline wall contribution:\n";
    ss << "its phase share times the phase's measured wall (assumes uniform\n";
    ss << "parallelism within the phase; '-' where no wall is registered).\n";
    ss << "'Wall' lines are single-timeline seconds registered by proveKernel's\n";
    ss << "phase barriers; attributed / wall = the sweep's effective parallelism.\n";
    ss << "The phase-2 wall is registered on both routes; a CUDA batch attributes\n";
    ss << "0 s to its phase-2 tree (the device runs outside every scope), so its\n";
    ss << "line then reads as the device wall alone.\n";
    ss << "\n";
    {
        char hdr[1024];
        const double p1w = st.phase1_wall_s;
        const double p3w = st.phase3_wall_s;
        const double p2w = st.phase2_wall_s;
        const double p1a = static_cast<double>(phase_ns[1]) / 1e9;
        const double p3a = static_cast<double>(phase_ns[3]) / 1e9;
        const double p2a = static_cast<double>(phase_ns[2]) / 1e9;
        char p1par[48] = "";
        char p3par[48] = "";
        char p2par[48] = "";
        if (p1w > 0.0)
            std::snprintf(p1par, sizeof(p1par), " = %.2fx parallel", p1a / p1w);
        if (p3w > 0.0)
            std::snprintf(p3par, sizeof(p3par), " = %.2fx parallel", p3a / p3w);
        if (p2w > 0.0)
            std::snprintf(p2par, sizeof(p2par), " = %.2fx parallel", p2a / p2w);
        std::snprintf(hdr, sizeof(hdr),
                      "Tracker calls (bursts)     : %lld\n"
                      "Summed burst lifetimes     : %.2f burst-seconds\n"
                      "Total attributed           : %.2f s   <- the 100%% of '%%attr'\n"
                      "Peak scope depth           : %d%s\n"
                      "Phase-1 tree               : %.2f s attributed / %.2f s wall%s\n"
                      "Phase-3 tree               : %.2f s attributed / %.2f s wall%s\n"
                      "Phase-2 tree (CPU route)   : %.2f s attributed / %.2f s wall%s\n\n",
                      static_cast<long long>(st.tracker_calls),
                      static_cast<double>(st.tracker_life_ns) / 1e9,
                      static_cast<double>(total_ns) / 1e9,
                      st.peak_open_depth,
                      st.peak_open_depth
                              > RTMeasurementParameters::RT_MAX_OPEN_DEPTH
                          ? "  (saturated past RT_MAX_OPEN_DEPTH)"
                          : "",
                      p1a, p1w, p1par,
                      p3a, p3w, p3par,
                      p2a, p2w, p2par);
        ss << hdr;
    }
    ss << "Parent > Section                                                     | seconds  | %attr  | %ph    | wall~s   | hits     | iter\n";
    ss << "---------------------------------------------------------------------+----------+--------+--------+----------+----------+----------\n";
    for (int idx : order) {
        const AggRow& r = st.rows[static_cast<std::size_t>(idx)];
        char name[2048];
        std::snprintf(name, sizeof(name), "%s", r.path.c_str());
        const double sec = static_cast<double>(r.self_ns) / 1e9;
        const double pct = (total_ns > 0)
            ? 100.0 * static_cast<double>(r.self_ns)
                    / static_cast<double>(total_ns)
            : 0.0;
        const int phase = phaseOfAggPath(r.path);
        const int64_t phaseTotal = phase_ns[phase];
        const double pctPhase = (phaseTotal > 0)
            ? 100.0 * static_cast<double>(r.self_ns)
                    / static_cast<double>(phaseTotal)
            : 0.0;
        const double phaseWall = (phase == 1) ? st.phase1_wall_s
                               : (phase == 3) ? st.phase3_wall_s
                               : (phase == 2) ? st.phase2_wall_s
                                              : 0.0;
        char wallCol[16];
        if (phaseWall > 0.0 && phaseTotal > 0) {
            std::snprintf(wallCol, sizeof(wallCol), "%8.2f",
                          phaseWall * static_cast<double>(r.self_ns)
                              / static_cast<double>(phaseTotal));
        } else {
            std::snprintf(wallCol, sizeof(wallCol), "%8s", "-");
        }
        char row[2400];
        if (r.iterations > 0) {
            std::snprintf(row, sizeof(row),
                          "%-68s | %8.2f | %6.2f | %6.2f | %s | %8lld | %lld\n",
                          name, sec, pct, pctPhase, wallCol,
                          static_cast<long long>(r.hits),
                          static_cast<long long>(r.iterations));
        } else {
            std::snprintf(row, sizeof(row),
                          "%-68s | %8.2f | %6.2f | %6.2f | %s | %8lld | %s\n",
                          name, sec, pct, pctPhase, wallCol,
                          static_cast<long long>(r.hits), "-");
        }
        ss << row;
    }

    // Atomic write-to-tmp + rename, same retry discipline as the per-call
    // snapshot: instrumentation never crashes the prover.
    std::error_code ec;
    const std::filesystem::path live(path);
    if (live.has_parent_path())
        std::filesystem::create_directories(live.parent_path(), ec);
    const std::string tmp_path = path + ".tmp";
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
        std::filesystem::rename(tmp_path, live, rename_ec);
        if (!rename_ec) return;
        std::this_thread::sleep_for(
            std::chrono::milliseconds(kRetryDelayMs));
    }
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
