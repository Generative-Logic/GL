<!--
Generative Logic: A deterministic reasoning and knowledge generation engine.
Copyright (C) 2025-2026 Generative Logic UG (haftungsbeschränkt).
Dual-licensed under the GNU Affero General Public License v3 or later
and a commercial license — see https://generative-logic.com/license.
-->

<!-- GL-AGENT-BANNER -->
> **Agent-oriented documentation.** This document is written for AI agents working with the GL codebase. Human readers: see the [paper](https://arxiv.org/abs/2508.00017) and the [README](../../../README.md). The document is intentionally dense, cross-linked, and weakness-explicit — agents thrive on that, humans usually don't.


# Runtime measurement infrastructure for `performElementaryLogicalStep` `[DRAFT]`

> Status: Living. Permanent, off-by-default instrumentation layer for
> diagnosing pathological logical blocks (LBs) whose single hash burst
> never finishes. Source-of-truth design document — every claim here
> should match the code in `src/infra/rt_tracker.hpp` / `src/infra/rt_tracker.cpp` and the
> `RT_SCOPE` call sites inside `performElementaryLogicalStep`.

---

## Purpose

Some LBs spend effectively forever inside a single hash burst — one call of
`prover.cpp::ExpressionAnalyzer::performElementaryLogicalStep`. We cannot
wait for the call to return to find out which LB is the offender and where
inside it the time goes. The infrastructure here lets a human, while the
run is still in progress, `ls .rt/` and immediately see (a) the list of
pathological LBs, and (b) for each of them a per-section timing table that
attributes the elapsed time to phases of `performElementaryLogicalStep`.

Two design choices come from that purpose:

- **Online refresh.** The per-LB file is rewritten in place every time a
 measurement boundary is crossed, so the table shown on disk always
 reflects the current state of the still-running call. The rewrite uses
 write-to-tmp + `std::filesystem::rename` so no reader ever sees a
 partially-written file.
- **Per-call scope.** All measurements belong to **one** call of
 `performElementaryLogicalStep`. Nothing is accumulated across calls,
 across outer iterations, or across worker threads. A new call of the
 same LB in a later burst overwrites the file from scratch.

The non-goal is production telemetry. The infrastructure is gated by a
compile-time macro that defaults to **off**, and when off it compiles to
zero instructions.

---

## Compile-time gate and tunables

All four entries live at the top of
[`parameters.hpp`](../GL_Quick_VS/GL_Quick/src/parameters.hpp) so they can
be edited as a group.

```cpp
#define RT_MEASUREMENT 0                                  // master gate
static constexpr int RT_TIME_TRIGGER_SECONDS = 120;       // dump threshold
static constexpr int RT_MIN_PERCENTAGE       = 0;         // table filter (0 = every row)
static constexpr int RT_MAX_SECTIONS         = 4096;      // per-call row storage (heap block per tracker)
```

| Symbol | Type | Default | What it controls |
|---|---|---|---|
| `RT_MEASUREMENT` | preprocessor macro | `0` | Master gate. `0` removes every `RT_SCOPE` call site at compile time; `1` activates them. Matches the `GL_DISINT_PROFILE` convention in [`compiler.hpp`](../GL_Quick_VS/GL_Quick/src/compiler.hpp). |
| `RT_TIME_TRIGGER_SECONDS` | `static constexpr int` | `120` | A `.rt/<chain>.log` file is written only after the current `performElem2` (hashburst) call has been running this many wall-clock seconds without returning. Calls that complete under the threshold leave no artefact. |
| `RT_MIN_PERCENTAGE` | `static constexpr int` | `0` | Sections whose self-time share is below this percentage of total elapsed are folded into a single trailing "(other sections each < N % of total)" line instead of appearing as their own row. `0` shows every row. |
| `RT_MAX_SECTIONS` | `static constexpr int` | `4096` | Capacity of the tracker's per-call `Section[]` row storage (one heap block per tracker, allocated in the constructor — a cold per-burst allocation, never on the per-scope path). |

The macro is a `#define` (rather than a `static constexpr bool`) on
purpose: every `RT_SCOPE` call site sits inside an `#if RT_MEASUREMENT`
block, so when the gate is `0` the preprocessor strips the calls and the
build is byte-identical to one where the class never existed.

---

## Lifecycle — one call, one tracker

The tracker is a stack-local object built at the top of
`performElem2` (the phase-2 hashburst executor; the original single-LB
orchestrator `performElementaryLogicalStep` was removed by the LB split).
Its destructor fires when the call returns. Nothing leaks across calls and
nothing crosses the thread boundary into the worker pool's join site. It
records meaningfully only with `disable_lb_split` set (one part per LB), so
the call is the LB's whole hashburst on one thread. Specifically:

- **Construction** records `t_start = steady_clock::now`, walks
 `body.parentMemory` to the root sentinel (empty `exprKey` + `nullptr`
 `parentMemory`, per the project conventions), and stores both the
 human-readable LB chain and a filesystem-safe sanitized form used as
 the eventual filename. No file is opened yet.
- **Section storage** is a `Section[RT_MAX_SECTIONS]` block owned by the
 tracker (`std::unique_ptr`, one allocation per tracker construction — at
 a few thousand rows the array no longer fits a worker's stack frame
 beside the door recursion). Each `Section` carries `{const char* label;
 int64_t self_ns; int hits; int64_t iterations; int parentIdx}` — no
 string allocation. A direct-mapped `(label pointer, parent) -> row`
 cache (`kRowCacheSize` 4096 slots) sits in front of the linear row scan
 so a scope open stays O(1) at thousands of rows.
- **Scope opening / closing** updates only the open-scope stack and the
 per-section running total. The exclusive-time model means each
 section's `self_ns` counts only the time the scope was the
 innermost-active one; time spent inside nested scopes is attributed
 to those nested rows.
- **Trigger and refresh** are checked on every scope close and on
 every `tracker.refreshIfTriggered` call inside long inner loops.
 Once `now − t_start` first crosses `RT_TIME_TRIGGER_SECONDS`, the
 table is written to `.rt/<chain>.log.tmp` and renamed over the live
 filename. Subsequent crossings overwrite the file with the latest
 snapshot.
- **Destruction** closes any still-open scope and attributes its
 remaining self-time. The destructor does not perform a final dump:
 the most recent `refreshIfTriggered` already wrote the latest
 state, and a call that finishes under the threshold leaves no
 artefact by design.

A new call to the same LB in a later burst starts a fresh tracker; the
older `.rt/<chain>.log` is overwritten the first time the new call
crosses the threshold.

---

## Code layout — `src/infra/`

The RT tracker and the existing hashburst dump both live under
`GL_Quick_VS/GL_Quick/src/infra/` (a small sub-tree dedicated to
diagnostic / instrumentation infrastructure that the main pipeline
calls *into* but does not otherwise interact with). The folder
collects code that is operationally independent of the prover's
algorithm — moving it out of `src/` keeps the top-level source
listing focused on the pipeline itself. The Makefile globs
`src/*.cpp`, `src/infra/*.cpp`, and `src/tests/*.cpp`; the Visual
Studio `.vcxproj` lists every file explicitly. Code in `src/infra/`
includes its own headers as `"infra/<name>.hpp"` (full path under
the `-Isrc` include root), and callers in `src/` do the same.

## Relationship to the hashburst dump (Rule 14)

The hashburst dump in
[`hashburst_dump.cpp`](../GL_Quick_VS/GL_Quick/src/infra/hashburst_dump.cpp)
remains untouched. It is a separate diagnostic tool, with separate
output (), separate targeting (a hard-coded
LB chain match), and separate semantics (state snapshot, not timing).
The RT infrastructure here adds a parallel mechanism for timing
diagnosis; the two never share files, gates, or code paths. Edits to
the hashburst dump still require explicit user consent per
the project conventions; edits to the RT infrastructure do not.

---

## Class shape

Two cooperating types in `namespace gl::rt_tracker`:

- **`RTTracker`** — the per-call state. Owns the fixed
 `Section sections_[RT_MAX_SECTIONS]` array, the open-scope index
 stack, the wall-clock start time, and the human-readable / sanitized
 chain strings. Non-copyable, non-movable; only stack construction
 is supported. Two constructors: a production one that reads
 `RTMeasurementParameters` for the trigger and the
 display-threshold defaults, and a test-only one that takes both
 values explicitly so unit tests in `src/tests/test_rt_tracker.cpp`
 can exercise the trigger logic in milliseconds rather than waiting
 the production 120 s.
- **`RTScope`** — RAII helper. Constructor calls
 `RTTracker::openSection_(label)`, destructor calls
 `RTTracker::closeSection_(index)`. Non-copyable. Holds only a
 pointer back to its tracker and the integer section index it
 opened.

**Exclusive-time stack model.** The tracker keeps an `open_stack_`
of section indices, one per nested `RTScope` currently alive, with
the innermost at the top. On every scope open or close (and at the
moment `refreshIfTriggered` decides to dump), the wall-clock delta
since the last event is added to the **top-of-stack** section's
`self_ns`, then `t_last_event_` is updated to *now*. Time spent
inside a nested scope therefore lands in that nested row, not in
the parent row — so parents reflect only their own work and the
column percentages do not double-count. A scope close validates
that the closing index equals the top-of-stack index; mismatch
asserts (per Rule 19) and surfaces RTScope objects being destroyed
out of construction order.

**Label storage.** Each section stores its label as a `const char*`
pointer, not a `std::string`. The `RT_SCOPE("LITERAL")` macro
mandates a string literal so the underlying storage lives in the
binary's read-only segment and outlives the tracker. The macro's
sole purpose for the call-site author is to make this contract
hard to misuse — a manually-constructed `RTScope` taking a
`someString.c_str` would also compile, but the convention is to
always go through the macro.

**Chain-to-filename sanitization.** `buildChainHuman_` walks
`body.parentMemory` to the root sentinel (Rule 12 chain match:
empty `exprKey` + `nullptr parent`), collects every `exprKey` into
a vector, reverses to root → leaf order, and joins with ` -> `. The
sentinel renders as the literal `(root)`. `sanitizeForFilename_`
takes that human string and replaces every filesystem-unsafe
character (`/ \: *? " < > | [ ],` plus whitespace) with
`_`. If the result exceeds `MAX_CHAIN_FILENAME_LEN` (120
characters, conservative so the full absolute path stays well
under Windows `MAX_PATH = 260` even from a deep worktree root),
the leading prefix is kept (still naming the outermost LB) and a
12-hex-char suffix derived from `std::hash<std::string>{}(chain)`
is appended for uniqueness.

**Atomic rewrite.** `refreshIfTriggered` checks `now - t_start_`
against `trigger_seconds_`; if at least the threshold has elapsed,
`writeSnapshot_` charges remaining time to the top-of-stack
section, sorts the section array by descending `self_ns`, writes
the rendered table to `.rt/<chain>.log.tmp` via `std::ofstream` +
explicit flush, then `std::filesystem::rename`s the temp file over
the live `.rt/<chain>.log`. `std::filesystem::rename` is atomic on
both Windows and POSIX for within-same-filesystem moves — a
concurrent reader sees either the previous complete snapshot or
the new complete snapshot, never a half-written file. The
function asserts (per Rule 19) on any I/O failure rather than
silently returning a sentinel.

**Write throttle — first write is exempt.** Between trigger firings,
`refreshIfTriggered` rate-limits the on-disk rewrite to at most once
per second (`now - t_last_refresh_ >= 1 s`) so inner-loop refreshes do
not thrash the file. The throttle applies only *after* the first write.
`t_last_refresh_` is seeded to construction time (`t_start_`), so without
an exemption the initial refresh computes ~0 ms-since-last and is wrongly
throttled — a tracker whose triggered lifetime is under a second would
then never write at all (and `~RTTracker`, which emits its final snapshot
only when `ever_dumped_`, would also stay silent). The gate is therefore
`ever_dumped_ && since_last_ms < 1000`: the first snapshot always lands,
subsequent ones throttle. (Before this fix the over-trigger unit test
passed only when a stale `.rt/` file happened to linger — masking the gap
on MSVC and surfacing it as a hard failure on the g++/Linux build.)

**Heap only at construction.** The row block and the row cache are
allocated once per tracker (see *Section storage* above), the open stack
is the fixed `int open_stack_[RT_MAX_OPEN_DEPTH]` array; the other
allocations are the two `std::string` chain members (constructor-set,
never resized again) and the in-memory `std::ostringstream` that builds
one snapshot before the rename — nothing allocates on the per-scope path.

## File format

Byte-accurate example of a populated `.rt/<sanitized>.log`:

```
LB chain (root -> leaf):
  (root)
  (AnchorPeano[N,i0,s,+,*,i1])
  (=[s(rec),zero])
  (=[v0,zero])

Total elapsed in this call : 245.30 s   (still running)
Trigger threshold          : 120 s
Display threshold          : 2 % of total

Section                                            | seconds | %total | hits | iter
---------------------------------------------------+---------+--------+------+-------
REQGEN_BATCH3_LOCAL_X_MAIL                         |  187.32 |  76.36 |    1 |      -
FIXPOINT_LOOP                                      |   43.21 |  17.62 |    1 |      1
PRE_FIXPOINT_MAIL_ABSORB                           |   12.45 |   5.08 |    1 |      -
---------------------------------------------------+---------+--------+------+-------
(8 sections each < 2 % of total; 2.32 s / 0.94 % combined)
```

Columns:

- **Section.** The label passed to `RT_SCOPE("...")`, left-padded
 to 50 columns. Labels are written in `SHOUT_SNAKE_CASE` so they
 group naturally when sorted alphabetically.
- **seconds.** `self_ns / 1e9`, fixed two decimals. Exclusive
 self-time — only the wall-clock spent inside that scope's own
 code, not in nested children.
- **%total.** `100 * self_seconds / total_elapsed`. Denominator is
 wall-clock elapsed in the current call, not the sum of recorded
 section times, so the column shows the section's true share of
 the call (and the rows do not need to sum to exactly 100 — any
 gap reflects time the tracker did not attribute to a scope,
 which is itself diagnostic information).
- **hits.** Number of times an `RTScope` opened with this exact
 label pointer during the current call. Typically 1 for top-level
 phase labels; higher for labels used inside loops.
- **iter.** Iteration count accumulated via `noteIterations(n)`
 while this scope was open. Rendered as `-` when the caller never
 called `noteIterations`. The combination `seconds / iter` gives
 the average per-iteration cost.

Rows are sorted by descending `seconds`. The `RT_MIN_PERCENTAGE`
filter is applied after sorting: any row whose `%total` is below
the threshold is folded into a single trailing
`(N sections each < M % of total; X.XX s / Y.YY % combined)` line
showing how many small rows were dropped and what fraction of the
call they collectively account for. A run with `RT_MIN_PERCENTAGE
= 0` shows every row and never emits the fold-down line.

The header is reprinted on every refresh. The `(still running)`
annotation in the elapsed line reflects the design: a tracker that
returned normally does not refresh again on the way out, so the
"final" file always reads as a mid-call snapshot. The final wall
time can be reconstructed by re-running with the trigger lowered.

## Where the scopes live

The elementary step is three phase helpers — see
[D-113](../40_decisions.md#d-113).
On the **live path** `proveKernel` drives them as three barriered sweeps (phase 1
for all active LBs → phase 2 for all → phase 3 for all,
[D-114](../40_decisions.md#d-114)).
The single-LB orchestrator `performElementaryLogicalStep` (which called all three
inline and declared the per-call tracker) became dead under the barriered driver
and has been **removed**. The per-call `RT_TRACKER_DECL(body)` is now homed in
`performElem2` (the hashburst executor), so one tracker measures one hashburst.
The phase helpers (`performElemPhase1`, `performElem2`, `performElemPhase2`,
`performElemPhase3`) bracket their work with `RT_SCOPE_HERE` (and the fixpoint
loop with `RT_NOTE_ITERATIONS_HERE`), which resolve the active tracker from the
thread-local `g_currentThreadTracker`.

> **RT scope (performElem2 home, [D-110]).** Because the
> tracker lives in `performElem2`, only the rows owned by `performElem2` (the
> `REQGEN_*` batches and `FIXPOINT_LOOP`) record. Rows owned by the other sweeps —
> `PRE_FIXPOINT_MAIL_ABSORB` (phase 1), `MERGE_FIRING_RECORDS` (phase-2 finalize),
> and the phase-3 rows — run with no tracker active and **do not record**. This is
> by design: RT diagnoses the hashburst, where a runaway LB spends its time;
> phases 1/3 are light. RT records only under `disable_lb_split` (one part per LB);
> without it an LB split into N parts spawns N `performElem2` calls that would
> collide on one `.rt/<chain>.log`. The labelled scopes, in execution order (with
> the owning phase helper):

| # | Label | Owner | Brackets |
|---|----------------------------------------|-------|----------|
| 1 | `PRE_FIXPOINT_MAIL_ABSORB` | `performElemPhase1` | `body.changedClassesThisStep.clear`, the workingMemory / externalStatements / intExternalStatements clears, and the pre-burst `standardProcessing` call. |
| — | `DOOR_CANONICALIZE` | `addExprToMemoryBlock` (nested) | The canonical door's `canonicalFormAtScope` call plus the raw-line / bridge / skip bookkeeping for a changed deposit — inside every absorb row's door call (both `standardProcessing` calls) and every direct deposit. |
| 2 | `REQGEN_CE_MODE` | `performElem2` | The `if (ceFilteringActive)` branch — single call to `generateEncodedRequestsStatic` at stump length 0. |
| 3 | `REQGEN_BATCH1_WORKING_MEMORY` | `performElem2` | Batch 1: `body.workingMemory` × local statements. Inside the `if (!body.workingMemory.encodedMap.empty)` guard, so the row only appears when the batch fires. |
| 4 | `REQGEN_BATCH23_NEW_THIS_BURST` | `performElem2` | Batches 2+3 folded: `body.overallHashMemory` under the mandatory-containment terms (local delta, OR mail arrival AND local statement). Unconditional. The retired `REQGEN_BATCH2_LOCAL_DELTA` / `REQGEN_BATCH3_LOCAL_X_MAIL` labels name this region in aggregates recorded before `D-298`. |
| 6 | `REQGEN_BATCH4_LOCAL_X_MAIL_SINGLES` | `performElem2` | Batch 4: `body.localHashMemory` × mail singles. Inside the dual `!empty` guard. |
| 7 | `REQGEN_BATCH5_LOCAL_HASH_DELTA` | `performElem2` | Batch 5: `body.localHashMemoryDelta`. Inside the `!empty` guard. |
| 8 | `STATIC_REQGEN_FIRE_EVAL` | `BurstSink::consume` (nested) | Request evaluation + firing (`checkLocalEncodedMemoryStatic`) for each emitted request that passes the dependency skip — the former `FIXPOINT_LOOP` work, folded into the sink when the separate evaluation pass was retired; the standalone `FIXPOINT_LOOP` label no longer exists. Nests inside whichever generation scope emitted the request. |
| — | `STATIC_REQGEN_STUMP_SEED` | `generateEncodedRequestsStatic` (nested) | Stump preparation (name-sort + comparability) and seed emission — a stump that already completes a whole key. |
| — | `STATIC_REQGEN_FILTER_SORT` | `generateEncodedRequestsStatic` (nested) | `filterIntEncodedStatements` over the statement universe + the name-only `stable_sort` — the fixed per-call cost the whole bucket shares. |
| — | `STATIC_REQGEN_GROW_SEARCH` | `generateEncodedRequestsStatic` (nested) | The DFS candidate enumeration: per-node request gates, key build, growth + record owner-set probes, `baseCandidates` accumulation. For an EMPTY obligatory stump (CE mode / whole-key batches) emission happens inside this scope. `iter` = the filtered statement count. |
| — | `STATIC_REQGEN_PAIRING_MERGE` | `generateEncodedRequestsStatic` (nested) | The PAIRING merge — every base candidate × every obligatory (mandatory) stump: scope fold, duplicate scan, `base ++ stump` name sort, `preEvaluateFromEncoded` whole-key probe, emit (dedup + arena copies). `iter` = the base-candidate count. Empty for stump length 0 and for the mandatory-containment mode, which has no merge at all. |
| — | `MERGE_FIRING_RECORDS` | `performElemPhase2` | The post-pass `applyFiringRecords` deposit merge. |
| 9 | `POST_FIXPOINT_MAIL_FLUSH` | `performElemPhase3` | The post-burst `standardProcessing` call, `sendMail`, and the three `body.mailOut.*.clear` calls. |
| 10| `REACT_TO_HYPO` | `performElemPhase3` | Just the `reactToHypo(body)` call. |
| 11| `END_OF_BURST_SANITIZE` | `performElemPhase3` | `sanitizeHashMemory`, `sanitizeToBeProved`, and the `body.pendingWipeScopes` drain. |

### Phase-1/3 atomic split (rt_investigation extension)

The tracker is additionally homed in `performElemPhase1` and
`performElemPhase3` (one `RT_TRACKER_DECL(body)` per call), so the phase-1
absorb and the phase-3 flush / sanitize record with full path attribution
in the cross-burst aggregate. Phase 2 keeps its `performElem2` home
untouched; the door scopes below no-op when the door runs with no tracker
active (the phase-2 finalize, the post-join barrier seams). New labelled
scopes, grouped by owner:

- **Phase helpers:** `PH1_CLAIM_LOAD`, `PH1_MAIL_PULL`, `PH3_CLAIM_LOAD`;
 under `END_OF_BURST_SANITIZE`: `SANITIZE_TOBEPROVED`,
 `DRAIN_DISPROVED_GOALS`, `DRAIN_DEAD_OR_BRANCHES`,
 `RETIRE_DUPLICATE_OR_COHORTS`, `FREEZE_RESOLVED_OR_BRANCHES`,
 `DRAIN_PENDING_OR_RELEASES`, `WIPE_SUBTREES`,
 `SWEEP_ANCESTOR_KNOWN_ROWS`.
- **`standardProcessing`:** `SP_ABSORB_ORIGINS_EXT` / `_INT` (with the
 nested `SP_ABSORB_EQORIGIN_SYNC`), `SP_ABSORB_STMTS_EXT` / `_INT`
 (`iter` = mailed statement count; nested `SP_ABSORB_STMT_ROW`,
 `SP_ABSORB_ROW_ORIGIN_MIN`, `SP_ABSORB_STAGE_EXTERNAL`),
 `SP_CLEAR_INTERNAL_MAIL`, `SP_APPLY_EQUI_CLASSES`,
 `SP_ENRICH_RECURSION_PRODUCTS`, `SP_CLEANUP_EXPRESSIONS`,
 `SP_DISCHARGE_CONTRADICTION`, `SP_DISCHARGE_TOBEPROVED`,
 `SP_DISCHARGE_CONTRA_SCOPES`, `SP_FILL_MAILOUT`,
 `SP_CLEAR_CHANGED_CLASSES`.
- **Deposit door (`addExprToMemoryBlock`):** `DOOR_ADD_EXPR` (self-time =
 gates + glue) with children `DOOR_CANONICALIZE` (pre-existing),
 `DOOR_GOAL_CHECK_NECESSITY`, `DOOR_PREPARE_INTEGRATION`,
 `DOOR_UPDATE_ADMISSION3`, `DOOR_CHECK_EQUIVALENCE`, `DOOR_DISINTEGRATE`,
 `DOOR_INSTALL_IMPLICATION`, `DOOR_GOALS_CHECK_NECESSITY`,
 `DOOR_STATEMENT_ROW` (per product statement; children
 `DOOR_ADD_STATEMENT`, `DOOR_ADMISSION_INTEGRATION`,
 `DOOR_ADMISSION_RECURSION`, `DOOR_ORDIS_MERGE`).
- **`addStatement`:** `ADDSTMT_ADD_EQUALITY`, `ADDSTMT_ADD_NEGATED_EQUALITY`,
 `ADDSTMT_COUNT_PATTERNS`, `ADDSTMT_UPDATE_EQUI_CLASSES`,
 `ADDSTMT_APPLY_TO_NEGEQ`.
- **`applyEquiClasses`:** the hook scopes of both class loops
 `AEC_HOOK_REJECTED_INTEGRATION`, `AEC_HOOK_ADMISSION_INTEGRATION`,
 `AEC_HOOK_REJECTED_ALGEBRA`, `AEC_HOOK_REJECTED_ORDIS`,
 `AEC_HOOK_ADMISSION_ALGEBRA`, `AEC_HOOK_COMPACT_IMPLICATIONS`
 (`AEC_PASS2` wraps the non-delta class loop); the statement batch
 `AEC_BATCH_TABLE` (once per apply), `AEC_BATCH_SCAN`,
 `AEC_BATCH_CLOSURE`, `AEC_BATCH_DRAIN` (with `AEC_BATCH_ORDIS` nested
 under it), and the per-row `AEC_NEGEQ_EXPANSION`.
- **`fillMailOut`:** `FMO_DELTA_ROW` (per delta row; nested
 `FMO_COPY_ORIGIN_ROWS`), `FMO_RELAY_DIRECT_DEPOSIT`; `iter` on the
 caller's `SP_FILL_MAILOUT` row = delta size.
- **`dischargeToBeProved`:** `DTP_DELTA_ROW` (per delta row),
 `DTP_ANCESTOR_GOAL_SCAN`; `iter` on `SP_DISCHARGE_TOBEPROVED` = delta
 snapshot size.
- **`disintegrateExpr2`:** `DISINT_PREPARE_CORE`, `DISINT_CORE`,
 `DISINT_PASSES_AB` (Pass A/B plus the rejection commits and the
 or/existence tail).
- **`cleanUpExpressions`** (under `SP_CLEANUP_EXPRESSIONS`): `CUE_VALIDITY`
 (one per deferred validity sweep; `iter` = the classes at that validity,
 probed per target-validity row in class order) with children
 `CUE_LOCAL_FILTER` and `CUE_LOCAL_DELTA_FILTER` (`filterIterations` over
 the local registry / its delta; `iter` = rows walked), `CUE_LOCAL_REBUILD`
 (packed-key set re-mint + the two local lists rebuilt from the kept rows),
 `CUE_STMTS_FILTER` (the statement registry walk; `iter` = rows walked),
 `CUE_STMTS_REBUILD`, and `CUE_WATERLINE_REPAIR` (only when a row was
 dropped).
- **`installExpandedImplication`** (under `DOOR_INSTALL_IMPLICATION`):
 `INSTALL_CANON_GATE` (self = `canonicalizeCompact`; nested
 `INSTALL_CANON_EXPAND` when K* is expanded instead, with
 `CANON_EXPAND_DEPS` / `CANON_EXPAND_DISINT` / `CANON_EXPAND_INSTALLS` /
 `CANON_EXPAND_REGISTER`), `INSTALL_DECOMPOSE`, the `addToHashMemory`
 targets `INSTALL_HASHMEM_OVERALL` (a status-3 install writes the
 incubator's `workingMemory` inside this scope as the second target of the
 same build — there is no separate working scope) / `INSTALL_HASHMEM_LOCAL`
 / `INSTALL_HASHMEM_DELTA`, `INSTALL_ORIGIN_ROW`
 (status-3 D-274 row), `INSTALL_INDEX_PAIR` (the `expandedImplications`
 pair) and `INSTALL_INDEX_COMPACT` (`appendCompactExpansion`).
- **`addToHashMemory`** (under whichever caller scope is open): `HM_MULTIPLY`
 (`multiplyImplication`; `iter` = copies), then per copy `HM_COPY` (self =
 arena setup + `idsRun` extraction) with children `HM_COPY_HISTORY` (the
 multiplied copy's origin rows), `HM_COPY_DISINT`, `HM_COPY_OWNER_MINT`
 (the three interner encodes), `HM_COPY_OWNERS` (the `copyOwners` run),
 `HM_ORIGINALS_CHAIN`, `HM_TRIGGERS` (`iter` = triggers),
 `HM_ADMISSION_NORMKEYS` (`makeNormalizedKeysForAdmission`, self = the
 qualification logic; children = the four passes `ADM_PASS_ORDIS2` /
 `ADM_PASS_INPUT_SLOT` / `ADM_PASS_ORDIS` / `ADM_PASS_REGULAR`, each holding
 the marker-variant installer's `ADM_SUBKEYS` (`iter` = permutations; per
 subkey `ADM_SK_ENCODE` / `ADM_SK_NORMKEY` / `ADM_SK_WRITE_SHORT` /
 `ADM_SK_WRITE_MERGE`), `ADM_WHOLEKEY` (`iter` = permutations; per
 permutation `ADM_WK_ENCODE` / `ADM_WK_NORMKEY` / `ADM_WK_VALUE_VARIANT` /
 `ADM_WK_KEYIDS` / `ADM_WK_LMV_FIELDS` / `ADM_LMV_APPEND` /
 `ADM_WHOLEKEY_OWNER`) and `ADM_REMARGS_BATCH`), `HM_REMARGS_MINT`,
 `HM_UPDATE_ADMISSION_MAP` (children `UAM_RENAME_CHAIN`,
 `UAM_PREPARE_INTEGRATION`, `UAM_ADMISSION_BLOB`, `UAM_REVISIT_ORDIS`),
 `HM_SHAPE_DETECT`, `HM_WHOLEKEY_RECORDS` (`iter` = permutations
 enumerated; per surviving permutation `HM_WK_ENCODE` (the
 `encodeExpression` twin per key element), `HM_WK_NORMKEY` (both normalized
 keys), `HM_WK_VALUE_VARIANT` (rename pairs + `replaceKeysScratch` + the
 value-id mint), `HM_WK_LMV_FIELDS` (remaining-arg ids + the `productOf`
 scan), `HM_LMV_APPEND`, `HM_WHOLEKEY_OWNER`), `HM_REMARGS_BATCH`,
 `HM_SUBKEYS` (`iter` = permutations; per subkey `HM_SK_ENCODE` /
 `HM_SK_NORMKEY` / `HM_SK_WRITE_SHORT` / `HM_SK_WRITE_MERGE`).
- **Rule-index staging** (D-333): the install
 sites' labels above now time the staging twins (`HM_LMV_APPEND` /
 `ADM_LMV_APPEND` contain `LMV_SERIALIZE` + `LMV_STAGE`; `HM_REMARGS_BATCH`
 / `ADM_REMARGS_BATCH` contain `RAB_SORT_DEDUP` + `RAB_STAGE`; the
 whole-key-owner and subkey-write labels hold the stage call); the cold
 writes happen once per window in `RULE_STAGING_FLUSH` (under
 `SP_FLUSH_RULE_STAGING` after the two absorbs, `AEC_FLUSH_RULE_STAGING`
 after the apply, or the unarmed `addToHashMemory`'s own exit), one child
 per section (`FLUSH_WHOLEKEY_OWNERS`, `FLUSH_ORIGINALS`,
 `FLUSH_COPY_OWNERS`, `FLUSH_REMARGS_OWNERS`, `FLUSH_LMV`, `FLUSH_SUBKEYS`,
 `FLUSH_REMARGS`; `iter` = staged keys of the section), each containing the
 store's in-place rebuild: `CSR_REBUILD_EMIT` (the one emit pass over the
 intact store, `iter` = keys) and `CSR_REBUILD_WRITE` (the backward tail
 write, `iter` = pool bytes behind the first produced key). The scratch
 twin used by the unit tests carries `CSR_REBUILD_EMIT` /
 `CSR_REBUILD_REFILL`.
- **Cold writers** (under their callers): `appendLmvIdsRecord` =
 `LMV_SERIALIZE` / `LMV_LOOKUP` / `LMV_WRITE`; `addOwnerToRun` (whole-key
 owners, copy owners, the originals chain, remaining-args edges) =
 `OWNER_LOOKUP` / `OWNER_MERGE` (`iter` = existing owners peeked and
 copied) / `OWNER_WRITE`; `addShortSubkeyOwner` = `SKS_LOOKUP_PEEK` /
 `SKS_BUILD` (`iter` = existing owners) / `SKS_WRITE`;
 `mergeSubkeySignatures` = `SKM_SIGNATURE` / `SKM_LOOKUP_PEEK` /
 `SKM_MERGE` (`iter` = existing signatures + owner pairs walked) /
 `SKM_WRITE`; `insertRemainingArgsNormKeyBatch` = `RAB_SORT_DEDUP` /
 `RAB_OWNER_EDGES` (`iter` = edges) / `RAB_MERGE_COUNT` (`iter` = existing
 records) / `RAB_WRITE` / `RAB_REVERSE_EDGES`.
- **The cold blob store** (`cold_hash_map.hpp`, under every writer):
 `CSR_MINT_KEY` (key append + index insert on a new key), `CSR_APPEND_RUN`
 (a new key's run), `CSR_REPLACE_RUN` (an existing key's run rewrite,
 containing `CSR_POOL_SHIFT_WRITE` — the blob-pool tail memmove plus the
 payload write, `iter` = pool bytes moved — and `CSR_BLOBSTARTS_SHIFT` —
 the blob-start column splice plus its suffix renumbering, `iter` = start
 entries behind the run), `CSR_RUNSTARTS_SHIFT` (every later key's run
 start renumbered, `iter` = keys behind).

`RT_MAX_SECTIONS` was raised to 192 for the deeper (label, parent) row space, then to 512 when the `HM_*` hash-memory install scopes (opened under every install caller) overflowed 192, then to 4096 (heap-backed, row cache) for the atomic install dissection, whose per-permutation / per-subkey / per-writer / per-store rows multiply by the four hash-memory targets and the admission passes; the original wording follows for the
space. The aggregate fold keys its rows through a path index
(`AggState::index`) and the dump's path buffer holds 2 KB, since the
install paths run past the former 255-character cap. The per-LB `.rt/<chain>.log` files are per-call and shared across
the three phase homes of one LB — a phase-1 file can be overwritten by the
same LB's phase-3 call once the trigger fires; the aggregate is the
authoritative report for this split.

The two surviving hashburst trap blocks bracket the live path (Rule 14, dump
unchanged — only the call-site location followed its bracketed code): ENTRY at
the top of `performElemPhase1`, EXIT at the bottom of `performElemPhase3`. (The
EARLY-EXIT trap that briefly lived in `performElemPhase2` was removed once the
deactivation deferral made it dead —
[D-114](../40_decisions.md#d-114).)
They sit *between* the labelled scopes and are not wrapped, and execute in well
under a millisecond each. **Parallel-split note:** the `performElem2` tracker is
per call, so at `splitCount > 1` each of an LB's N parts would build its own
tracker and they would overwrite one another's `.rt/<chain>.log`. RT is therefore
used with `disable_lb_split` (`splitCount == 1`), where each LB has exactly one
`performElem2` call and one tracker.

The arena (`TypedArena<IntEncodedExpr> exprArena`) and the request buffer
(`std::vector<StaticRequest> reqBuf`) are now declared in `performElemPhase2`
inside the `if (body.isActive)` gate and passed to `performElem2` (caller-owned
so each parallel executor gets its own). Their declaration microseconds appear
in the "unattributed" gap between section sums and total wall-clock — a
deliberate design choice, not an omission.

## Parent-keyed sections

Sections are keyed by **(label, parent)** — the parent being the section
that was innermost-open when the label first opened under it. One label
opened under two different enclosing scopes therefore yields two
separately-attributed rows, rendered `PARENT > LABEL` in every table.
This is what keeps the nested `STATIC_REQGEN_*` interior scopes from
collapsing the per-batch attribution: `STATIC_REQGEN_PAIRING_MERGE`
under `REQGEN_BATCH3_LOCAL_X_MAIL` and under
`REQGEN_BATCH4_LOCAL_X_MAIL_SINGLES` stay distinct rows, and a batch's
inclusive total is its own row plus its children's rows.

## The cross-burst aggregate

The per-LB `.rt/<chain>.log` files are per-call snapshots: trigger-gated,
overwritten by the LB's next burst — they cannot give run totals. The
**aggregate** closes that gap: every `RTTracker` destructor folds its
sections into one process-wide table keyed by the FULL open-scope path
(root → leaf labels joined with " > ", so a shared interior label stays
attributed to the batch that owns the chain) — trigger-independent, so
EVERY burst contributes — and
`main.cpp` writes it once at end of batch to `.rt/_aggregate_<tag>.log`
(`gl::rt_tracker::dumpRtAggregate`). The header carries the tracker-call
count and the summed burst lifetimes ("burst-seconds"; bursts run in
parallel, so the sum exceeds wall-clock); rows carry cumulative seconds,
% of total attributed time, hits, and iteration counts.

**Wall lines and the `wall~s` column.** `proveKernel` registers each iteration's barrier-to-barrier wall for all three phases (`addRtPhaseWallSeconds` — phases 1 and 3 at the post-phase-3 barrier, phase 2 at the phase-2 barrier on both routes). The header prints per phase tree `attributed / wall = N× parallel`, the sweep's effective parallelism; a CUDA batch attributes 0 s to its phase-2 tree (the device runs outside every scope), so that line reads as the device wall alone. `%ph` is a row's share of its own phase tree and `wall~s` is that share times the phase's wall — the row's estimated single-timeline contribution under uniform parallelism within the phase.

The aggregate is **split-safe**: at `splitCount > 1` each of an LB's N
parts builds its own tracker and their sections sum correctly in the
aggregate (the per-LB *files* still collide — that caveat is unchanged).
An aggregate-only measurement run therefore does NOT need
`disable_lb_split` and can profile the production-shaped run.

The aggregate is additive-only, mutex-guarded on the cold per-burst
destructor path, and read by nothing in the prover — a dump-only
telemetry sink, never a proof input (the Rule 16 discipline). It is the
one documented exception to I-59's per-call scope.

## Operational use

**Flipping the gate on.** Edit `parameters.hpp`:

```cpp
#define RT_MEASUREMENT 1                                  // was 0
```

and set `disable_lb_split: true` in the batch's `prover_parameters` (RT
records only with one part per LB — see the performElem2-home note above).
Then rebuild (`MSBuild.exe GL_Quick.sln -p:Configuration=Release` on
Windows; `make -j$(nproc)` on Linux). Every `RT_SCOPE_HERE` call site in
`performElem2` becomes active; the rest of the binary is byte-identical to
the production build apart from the tracker's constructor / destructor calls.

**Probing with a low trigger.** During investigation, lower the trigger
to capture shorter LBs (e.g. 5 seconds):

```cpp
static constexpr int RT_TIME_TRIGGER_SECONDS = 5;         // was 120
```

Then rebuild and re-run. Every LB whose `performElem2` call (its hashburst)
exceeds 5 seconds writes a snapshot to `.rt/<sanitized-chain>.log`.

**Watching mid-run.** From a separate terminal:

```bash
watch -n 1 'ls -la .rt/ ; echo ; cat .rt/*.log 2>/dev/null | head -40'
```

The `.rt/` listing grows as bad LBs cross the threshold; each file's
header banner shows `(still running)` and the table refreshes on every
scope boundary inside the LB's call.

**Drilling deeper.** Once a phase row dominates a file (e.g.
`REQGEN_BATCH3_LOCAL_X_MAIL` at 80 % of total), add a nested set of
`RT_SCOPE` calls inside that phase's source block to split its
self-time across sub-phases. Per the exclusive-time stack model the
new inner rows show their own work and the original outer row drops to
just the time outside any nested scope. The first inner pass typically
identifies the offender within one or two iterations.

**Restoring production.** Revert the `parameters.hpp` edits
(`RT_MEASUREMENT = 0` and `RT_TIME_TRIGGER_SECONDS = 120`) and set
`disable_lb_split` back to `false`, rebuild, verify with
`gl_quick.exe --unit-tests` that all unit tests pass, and confirm `.rt/`
stays empty across a normal run.

**Clean-slate per session.** `main.py` wipes `.rt/*.log` at start so
each pipeline run begins with no leftover files. The C++ side never
wipes the folder; sibling `gl_quick.exe` invocations within the same
`main.py` session do not overwrite each other's empty state.

---

## Weaknesses

### Suspected fragility

- **Per-phase trackers, per-call files.** Since the rt_investigation extension, all three phase helpers home a tracker (`performElemPhase1`, `performElem2`, `performElemPhase3`), so phase-1/3 rows record with full path attribution. The per-LB `.rt/<chain>.log` files are still per-call and one LB's three phase trackers share one filename — the aggregate is the authoritative cross-phase report. The `disable_lb_split` caveat remains for phase-2 per-LB FILES only; the aggregate and the phase-1/3 trackers are split-safe (phases 1/3 are always one call per LB).
- **Section labels are `const char*` literals.** The tracker stores label
 pointers, not strings. A dynamic label would dangle once its source
 string goes out of scope. The `RT_SCOPE` macro takes a string literal
 exactly to make this hard to misuse, but the class accepts any
 `const char*` — a future caller could pass `someStr.c_str` and read
 freed memory at dump time.
- **`RT_MAX_SECTIONS = 4096` is a hard cap.** Adding deeply nested scopes
 (the design anticipates two or three levels deep) can in principle
 exhaust the array. The class asserts on overflow per
 the project conventions; no silent dropping.
- **Cross-call comparison is not supported.** Each `performElem2`
 call rewrites its file from scratch; if the same LB hangs for two
 bursts in a row, only the second burst's table survives. This is by
 design (per-call scope is the contract) but means a longitudinal
 comparison requires `cp .rt/<chain>.log .rt/<chain>__burst1.log` by
 hand between bursts.

### Not exercised by tests

- The class itself ships with positive + negative unit tests under
 `src/tests/test_rt_tracker.cpp`, and `disable_lb_split` has a
 default-false unit test in `src/tests/test_lb_split.cpp`. The
 `performElem2` call sites are exercised by hand (build with
 `RT_MEASUREMENT = 1` and `disable_lb_split: true`, lower the trigger, run
 `main.py`, inspect `.rt/`). There is no automated integration test
 verifying that the expected section labels appear, because the call
 boundaries are pure source-level annotations and any
 drift is caught by ordinary review.

---

<!-- GL-PAGE-FOOTER -->
**Generative Logic** — © 2025-2026 Generative Logic UG (haftungsbeschränkt). Dual-licensed under the [GNU Affero General Public License v3 or later](https://www.gnu.org/licenses/agpl-3.0.html) and a [commercial license](https://generative-logic.com/license). Source: [github.com/Generative-Logic/GL](https://github.com/Generative-Logic/GL) · Paper: [arxiv.org/abs/2508.00017](https://arxiv.org/abs/2508.00017)
