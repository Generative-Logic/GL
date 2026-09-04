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

#pragma once

#include "parameters.hpp"

#include <chrono>
#include <cstdint>
#include <memory>
#include <string>

namespace gl {

    struct Memory;

    /// @brief Permanent off-by-default runtime-measurement infrastructure.
    ///
    /// @details
    /// Owns one stack-local `RTTracker` per `RT_TRACKER_DECL` scope. After the LB
    /// split removed the single-LB orchestrator, the tracker is homed in
    /// `performElem2` (the hashburst executor), so one tracker measures one
    /// hashburst. It records only with `disable_lb_split` set (one part per LB),
    /// where that call is the LB's whole hashburst on one thread — see
    /// `docs/agentic_swdd/_meta/rt_measurement.md`.
    /// The tracker walks the LB chain to the root sentinel in its
    /// constructor, records `t_start`, and from that point on attributes
    /// elapsed wall-clock time to labelled sections opened by
    /// `RTScope` RAII helpers. Once the call has been running longer
    /// than `RTMeasurementParameters::RT_TIME_TRIGGER_SECONDS` without
    /// returning, the tracker writes a per-LB textual table to
    /// `.rt/<sanitized-chain>.log` via atomic write-to-tmp + rename,
    /// and refreshes the file on every subsequent scope boundary so a
    /// reader inspecting `.rt/` mid-call sees the latest state.
    ///
    /// @par Lifecycle
    /// One tracker per elementary step. Nothing is
    /// shared with other calls, other threads, or other outer iterations
    /// of `ExpressionAnalyzer::prove`. Per `I-59`.
    ///
    /// @par Relationship to the hashburst dump
    /// Separate file (`.debug/hashburst_trace.txt` vs `.rt/<chain>.log`),
    /// separate gate (`hashburst_dump::isTargetLB` is a hard-coded chain
    /// match; this infrastructure is a wall-clock-threshold trigger),
    /// separate code path. The hashburst dump is protected by Rule 14;
    /// edits here do not.
    ///
    /// @par Compile-time gate
    /// All RT call sites in the elementary step's phase helpers are guarded by
    /// `#if RT_MEASUREMENT`. When the macro is `0`, every call site is
    /// stripped at preprocess time and the binary is byte-identical to
    /// one that never had the infrastructure. The class definition itself
    /// is compiled unconditionally so unit tests in `src/tests/` can
    /// exercise it regardless of the gate.
    ///
    /// @see docs/_meta/rt_measurement.md
    /// @see D-95
    /// @see I-59
    namespace rt_tracker {

        class RTTracker;

        /// @brief RAII helper that opens a labelled section on construction
        ///        and closes it on destruction.
        ///
        /// @details
        /// Sections nest. The enclosing `RTTracker` keeps a stack of
        /// currently-open scopes; on every open/close, the wall-clock
        /// delta since the previous event is attributed to the
        /// top-of-stack section (exclusive self-time model). Time spent
        /// inside a nested scope is attributed to that nested row, not
        /// to its parent — so the parent's row reflects only its own
        /// code, and percentages do not double-count.
        ///
        /// @par Lifetime
        /// Must not outlive its `RTTracker`. The class is non-copyable
        /// and non-movable; only stack construction is supported.
        ///
        /// @see RTTracker
        /// @see RT_SCOPE
        class RTScope {
        public:
            /// @brief Open a labelled section.
            ///
            /// Appends a fresh `{label, 0, 0, 0}` row to the tracker's
            /// sections array (or finds an existing row with the same
            /// label pointer and increments its `hits` counter), then
            /// pushes the row's index onto the tracker's open-scope
            /// stack. Asserts (per Rule 19) if the array is full.
            ///
            /// @param tracker The owning tracker.
            /// @param label   String literal — the tracker stores the
            ///                pointer, so the storage must outlive the
            ///                tracker. The `RT_SCOPE` macro enforces
            ///                this by accepting only literals.
            RTScope(RTTracker& tracker, const char* label);

            /// @brief Close the section, attributing remaining self-time.
            ~RTScope();

            RTScope(const RTScope&)            = delete;
            RTScope& operator=(const RTScope&) = delete;

        private:
            RTTracker* tracker_;
            int        section_index_;
        };

        /// @brief Per-call timing tracker for the elementary step.
        ///
        /// @details
        /// Construct one stack-local instance per call. Open `RTScope`
        /// RAII blocks around the function's existing phases (mail
        /// absorb, request-generation batches, the fixpoint loop,
        /// post-fixpoint flush, hypo reaction, end-of-burst sanitize).
        /// Inside long inner loops, call `refreshIfTriggered()` every
        /// N iterations so the on-disk table keeps up with the
        /// in-flight call even when no scope is closing.
        ///
        /// @par Storage
        /// Fixed `Section sections_[RT_MAX_SECTIONS]` array — no heap.
        /// Matches the `ChunkPool` static-`char[]` convention pinned
        /// by `I-13`.
        ///
        /// @par Filename
        /// The constructor walks `body.parentMemory` to the root
        /// sentinel (empty `exprKey` + `nullptr` parent, per Rule 12),
        /// joins exprKeys with `__` in root → leaf order, replaces
        /// every filesystem-unsafe character (`/ \\ : * ? " < > | ( ) [
        /// ] , space`) with `_`, and caps the final path so that
        /// `.rt/<sanitized>.log` fits within `MAX_PATH`. When the
        /// human-readable chain is too long, the suffix is replaced
        /// by a short hash so two different chains never collide on
        /// the same filename.
        ///
        /// @par Atomic rewrite
        /// `refreshIfTriggered()` first writes the table to
        /// `.rt/<chain>.log.tmp`, then `std::filesystem::rename`s the
        /// temporary file over the live one. A reader inspecting
        /// `.rt/<chain>.log` mid-call therefore sees either the previous
        /// complete snapshot or the new complete snapshot — never a
        /// half-written file.
        ///
        /// @see RTScope
        /// @see I-59
        class RTTracker {
        public:
            using Clock = std::chrono::steady_clock;

            /// @brief Construct a tracker on the production thresholds.
            ///
            /// Uses `RTMeasurementParameters::RT_TIME_TRIGGER_SECONDS`
            /// and `RTMeasurementParameters::RT_MIN_PERCENTAGE`. Walks
            /// the LB chain, builds the human-readable and filename
            /// forms, records `t_start`. Does not open a file.
            ///
            /// @param body The LB the enclosing
            ///        elementary step is processing.
            ///        The reference is borrowed for the constructor body
            ///        only; the tracker does not retain it.
            explicit RTTracker(const Memory& body);

            /// @brief Construct a tracker on caller-supplied thresholds.
            ///
            /// Test-only entry. Production code uses the one-argument
            /// constructor exclusively.
            ///
            /// @param body              See the production constructor.
            /// @param triggerSeconds    Override for
            ///                          `RT_TIME_TRIGGER_SECONDS`.
            /// @param minPercentage     Override for
            ///                          `RT_MIN_PERCENTAGE`.
            RTTracker(const Memory& body, int triggerSeconds, int minPercentage);

            /// @brief Close any still-open scope and stop measuring.
            ///
            /// Does not perform a final dump: the most recent
            /// `refreshIfTriggered()` (whether from a scope close or an
            /// inner-loop refresh) has already written the latest
            /// snapshot, and a call that finishes under the threshold
            /// leaves no artefact by design.
            ~RTTracker();

            RTTracker(const RTTracker&)            = delete;
            RTTracker& operator=(const RTTracker&) = delete;

            /// @brief Stamp the currently-open scope with an iteration
            ///        count.
            ///
            /// Adds `n` to the top-of-stack section's `iterations` field.
            /// Used by inner loops (the fixpoint loop in particular) to
            /// expose an average-per-iteration figure on the dumped
            /// table. Asserts (per Rule 19) if no scope is currently
            /// open.
            ///
            /// @param n Number of inner-loop iterations to accumulate.
            void noteIterations(int64_t n);

            /// @brief Refresh the on-disk table if the trigger has fired.
            ///
            /// Called automatically on every scope close. Also called
            /// manually inside long inner loops every N iterations so
            /// the on-disk table tracks an in-flight call even when no
            /// scope is closing. A no-op if the call has not yet been
            /// running longer than the trigger threshold.
            void refreshIfTriggered();

            /// @brief Human-readable LB chain (root → leaf).
            ///
            /// Test accessor; the production caller never reads this.
            const std::string& chainHuman() const { return chain_human_; }

            /// @brief Sanitized chain used to derive the filename.
            ///
            /// Test accessor; the production caller never reads this.
            const std::string& chainFilename() const { return chain_filename_; }

        private:
            friend class RTScope;
            friend class RTScopeHere;

            /// Push a new open scope onto the stack; return the section
            /// index. Allocates the row at the array's tail if no
            /// prior row carries the same `label` pointer (label
            /// pointer equality, not string equality — the macro
            /// guarantees identical string literals share an address).
            int  openSection_(const char* label);

            /// Pop the top-of-stack open scope, attributing remaining
            /// self-time to the section at `sectionIndex`. Asserts on
            /// stack mismatch (per Rule 19).
            void closeSection_(int sectionIndex);

            /// Add the wall-clock delta since the last event to the
            /// top-of-stack section's `self_ns`. Called from every
            /// `openSection_` / `closeSection_` boundary.
            void chargeElapsedToTop_();

            /// Write the latest table to `.rt/<chain>.log.tmp`, then
            /// `std::filesystem::rename` over the live file. Creates
            /// `.rt/` if absent.
            ///
            /// @param finished When true, header reads
            ///        `Total elapsed in this call : X.XX s   (finished)`
            ///        instead of `(still running)`. Called once from the
            ///        destructor for any tracker that ever crossed the
            ///        trigger.
            void writeSnapshot_(bool finished = false);

            /// Fold this call's sections into the process-wide aggregate
            /// (see `resetRtAggregate` / `dumpRtAggregate`). Called once
            /// from the destructor, after the final self-time charge,
            /// for EVERY tracker — trigger-independent.
            void accumulateAggregate_();

            /// Build the root → leaf chain string of `body.exprKey`s.
            static std::string buildChainHuman_(const Memory& body);

            /// Convert the human-readable chain into a filesystem-safe
            /// filename (no extension). Caps length and appends a short
            /// hash on overflow.
            static std::string sanitizeForFilename_(const std::string& chain);

            struct Section {
                const char* label;
                int64_t     self_ns;
                int         hits;
                int64_t     iterations;   // byte / entry tallies overflow int32 per burst
                /// Index of the section that was innermost-open when this
                /// one first opened, or -1 at tracker top level. Sections
                /// are keyed by (label, parentIdx), so one label opened
                /// under two different parents yields two rows — the
                /// per-parent attribution the REQGEN batch split needs.
                int         parentIdx;
            };

            /// Row storage — `RT_MAX_SECTIONS` rows on the heap, allocated
            /// once per tracker (a cold per-burst allocation, never on the
            /// per-scope path): at a few thousand rows the array no longer
            /// fits a worker's stack frame beside the door recursion.
            std::unique_ptr<Section[]> sections_;
            int              section_count_;
            /// Direct-mapped (label pointer, parent index) -> row index cache
            /// in front of the linear row scan, so a scope open stays O(1)
            /// when a tracker carries thousands of rows. A miss falls back to
            /// the scan and refills the slot; -1 = empty.
            static constexpr int kRowCacheSize = 4096;   // power of two
            std::unique_ptr<int[]> row_cache_;
            int              open_stack_[RTMeasurementParameters::RT_MAX_OPEN_DEPTH];
            int              open_depth_;
            /// Count of currently-open VIRTUAL scopes — opens that arrived
            /// with the stack at RT_MAX_OPEN_DEPTH and were saturated (not
            /// pushed; their time folds into the innermost tracked section).
            int              virtual_depth_;
            /// Deepest nesting this call ever reached, virtual frames
            /// included — the evidence row for how deep a re-entrant
            /// production chain went.
            int              peak_open_depth_;
            Clock::time_point t_start_;
            Clock::time_point t_last_event_;
            Clock::time_point t_last_refresh_;
            std::string      chain_human_;
            std::string      chain_filename_;
            int              trigger_seconds_;
            int              min_percentage_;
            int              hashburst_index_;
            bool             ever_dumped_;
        };

        /// @brief Hashburst index for the current `prove()` outer iteration.
        ///
        /// Set by `prove()` immediately before each `proveKernel` call so
        /// every `RTTracker` constructed during that burst captures the
        /// same index. Worker threads read it during their
        /// elementary-step phase sweeps; main thread writes it
        /// between bursts (no concurrent write while workers run).
        /// Initialised to `-1` so any read before `prove()` first sets it
        /// is identifiably "not in a burst".
        extern int g_currentHashburstIndex;

        /// @brief Thread-local pointer to the currently-active tracker.
        ///
        /// Set by `RTTracker`'s constructor, cleared by its destructor.
        /// Lets the `RT_SCOPE_HERE` / `RT_REFRESH_HERE` macros (used by
        /// inner request-generation functions called from
        /// the elementary step's phase helpers) find the per-call tracker
        /// without threading it through every function signature.
        /// `nullptr` when no tracker is active on this thread — the
        /// "here" macros tolerate the null pointer and no-op.
        extern thread_local RTTracker* g_currentThreadTracker;

        /// @brief RAII helper that opens a labelled section on the
        ///        currently-active thread tracker, if any.
        ///
        /// @details Symmetric to `RTScope` but reads the tracker from
        /// `g_currentThreadTracker` instead of taking it by reference.
        /// When no tracker is active on this thread (e.g. called outside
        /// any `RT_TRACKER_DECL` scope), construction is a no-op.
        class RTScopeHere {
        public:
            explicit RTScopeHere(const char* label);
            ~RTScopeHere();
            RTScopeHere(const RTScopeHere&) = delete;
            RTScopeHere& operator=(const RTScopeHere&) = delete;
        private:
            RTTracker* tracker_;
            int        section_index_;
        };

        /// @brief Reset the process-wide RT aggregate to empty.
        ///
        /// @details
        /// The aggregate is the cross-burst sink: every `RTTracker`
        /// destructor folds its per-call sections into a process-wide
        /// table keyed by the full open-scope path (root → leaf labels
        /// joined with " > "), regardless of whether
        /// the call ever crossed the dump trigger — so the aggregate
        /// covers EVERY burst, while the per-LB `.rt/<chain>.log` files
        /// remain trigger-gated per-call snapshots. This is the one
        /// documented exception to I-59's per-call scope: the aggregate
        /// is additive-only, mutex-guarded on the cold destructor path,
        /// read by nothing in the prover — a dump-only telemetry sink,
        /// never a proof input (Rule 16 discipline).
        ///
        /// Test-and-startup entry; production code calls it never (the
        /// aggregate lives for the process, one batch per process).
        void resetRtAggregate();

        /// @brief Write the process-wide RT aggregate table to @p path.
        ///
        /// @details
        /// Renders every full-path row with cumulative seconds,
        /// hits, and iteration counts, sorted by descending seconds,
        /// plus a header carrying the number of contributing tracker
        /// calls (bursts) and their summed lifetimes ("burst-seconds" —
        /// bursts run in parallel, so the sum exceeds wall-clock).
        /// Percentages are of total attributed nanoseconds. Atomic
        /// write-to-tmp + rename like the per-call snapshot; failures
        /// retry then return silently (instrumentation never crashes
        /// the prover).
        ///
        /// @param path Destination file path (caller chooses folder).
        void dumpRtAggregate(const std::string& path);

        /// @brief Accumulate one iteration's phase wall-clock into the
        ///        aggregate (single-timeline seconds, NOT worker-seconds).
        ///
        /// @details
        /// Called from `proveKernel` once per iteration for phase 1 and
        /// phase 3 with that iteration's barrier-to-barrier wall time.
        /// `dumpRtAggregate` divides each phase tree's attributed
        /// worker-seconds by this wall sum to print the effective
        /// parallelism, so the table defines exactly what its seconds
        /// mean against the run's single timeline.
        ///
        /// @param phase   1, 2 or 3 — the three barriered phases; phase 2
        ///                registers on both routes (the CUDA route attributes
        ///                nothing to its tree, so the line reads as device wall).
        /// @param seconds Wall-clock seconds of that phase's sweep this
        ///                iteration.
        void addRtPhaseWallSeconds(int phase, double seconds);

        /// @brief Refresh the on-disk snapshot via the thread tracker.
        ///
        /// Reads `g_currentThreadTracker`; no-op when null. Used inside
        /// inner request-generation loops where threading a tracker
        /// reference through every function signature would be costly.
        inline void rtRefreshHere() {
            if (g_currentThreadTracker) {
                g_currentThreadTracker->refreshIfTriggered();
            }
        }

        /// @brief Annotate the currently-active thread tracker's open scope
        ///        with an iteration count (no-op when no tracker is active).
        ///
        /// @details Symmetric to `rtRefreshHere` but for `noteIterations`.
        /// Used by the fixpoint loop after the performElem phase-split moved it
        /// out of the function that owns the named `_rtTracker_` local into a
        /// phase helper; the helper reads the tracker from
        /// `g_currentThreadTracker` (set by the orchestrator's
        /// `RT_TRACKER_DECL`) instead.
        ///
        /// @param n Number of inner-loop iterations to accumulate.
        inline void rtNoteIterationsHere(int64_t n) {
            if (g_currentThreadTracker) {
                g_currentThreadTracker->noteIterations(n);
            }
        }

    } // namespace rt_tracker

} // namespace gl

// ---------------------------------------------------------------------------
// Call-site macros — strip to nothing when RT_MEASUREMENT == 0.
// ---------------------------------------------------------------------------
//
// Convention: callers declare the tracker once per function with
// `RT_TRACKER_DECL(body)`, which introduces a local variable named
// `_rtTracker_`. All other macros expand to references to that name.
// When `RT_MEASUREMENT` is 0, every macro expands to `((void)0)` so
// the local variable is never declared and the caller's function body
// is byte-identical to a build that never had the infrastructure.

#define GL_RT_CONCAT2(a, b) a##b
#define GL_RT_CONCAT(a, b)  GL_RT_CONCAT2(a, b)

#if RT_MEASUREMENT

    /// Declare the per-call tracker. Place once at the top of the instrumented
    /// function (now `performElem2`, the hashburst executor; records under
    /// `disable_lb_split` — see `docs/agentic_swdd/_meta/rt_measurement.md`).
    /// Subsequent `RT_SCOPE` / `RT_REFRESH` /
    /// `RT_NOTE_ITERATIONS` references resolve to this variable.
    #define RT_TRACKER_DECL(body) \
        ::gl::rt_tracker::RTTracker _rtTracker_((body))

    /// Bracket a labelled section. Place inside a `{ }` block to
    /// control the section's lifetime; the RAII destructor closes the
    /// section at the closing brace. The label must be a string
    /// literal so the underlying pointer survives the call.
    #define RT_SCOPE(label) \
        ::gl::rt_tracker::RTScope GL_RT_CONCAT(_rt_scope_, __LINE__)(_rtTracker_, (label))

    /// Refresh the on-disk table if the trigger has fired. Use inside
    /// long inner loops at a sample rate (e.g. every 64 iterations).
    #define RT_REFRESH() _rtTracker_.refreshIfTriggered()

    /// Annotate the currently-open scope with an iteration count.
    #define RT_NOTE_ITERATIONS(n) _rtTracker_.noteIterations((n))

    /// Bracket a labelled section using the thread-local tracker
    /// (no-op if no tracker is active on this thread). Use in inner
    /// functions that don't have access to the named `_rtTracker_`
    /// local — they get the tracker from `g_currentThreadTracker`.
    #define RT_SCOPE_HERE(label) \
        ::gl::rt_tracker::RTScopeHere GL_RT_CONCAT(_rt_scope_here_, __LINE__)((label))

    /// Refresh the on-disk snapshot via the thread-local tracker
    /// (no-op if no tracker is active on this thread). Use inside
    /// inner request-generation loops at a "per new seed" cadence.
    #define RT_REFRESH_HERE() ::gl::rt_tracker::rtRefreshHere()

    /// Annotate the open scope with an iteration count via the
    /// thread-local tracker (no-op if none active). The `_HERE` form of
    /// `RT_NOTE_ITERATIONS` for phase helpers that don't own `_rtTracker_`.
    #define RT_NOTE_ITERATIONS_HERE(n) ::gl::rt_tracker::rtNoteIterationsHere((n))

#else

    #define RT_TRACKER_DECL(body) ((void)0)
    #define RT_SCOPE(label)       ((void)0)
    #define RT_REFRESH()          ((void)0)
    #define RT_NOTE_ITERATIONS(n) ((void)0)
    #define RT_SCOPE_HERE(label)  ((void)0)
    #define RT_REFRESH_HERE()     ((void)0)
    #define RT_NOTE_ITERATIONS_HERE(n) ((void)0)

#endif
