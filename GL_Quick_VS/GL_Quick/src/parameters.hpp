/* Generative Logic : A deterministic reasoning and knowledge generation engine.
 Copyright(C) 2025-2026 Generative Logic UG(haftungsbeschr�nkt)

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

// ---------------------------------------------------------------------------
// RT measurement compile-time gate (permanent instrumentation; off by default)
// ---------------------------------------------------------------------------
// When 1, the hashburst executor `performElem2` opens an RAII timing tracker
// for the duration of one call (homed there since the LB split removed the
// single-LB orchestrator — see `_meta/rt_measurement.md`). It records only with
// `disable_lb_split` set (one part per LB), so each call measures the LB's whole
// hashburst on one thread. If the call exceeds `RT_TIME_TRIGGER_SECONDS`, a
// per-LB textual table is rewritten under `.rt/<sanitized-LB-chain>.log`
// online (atomic write-tmp + rename) until the call returns. Only sections
// whose self-time share is at least `RT_MIN_PERCENTAGE` of total elapsed
// appear in the table. Measurements are scoped to **one** call — never
// accumulated across calls or outer iterations. When 0, every call site
// compiles to nothing and the tracker class itself is untouched (it still
// compiles unconditionally so unit tests can exercise it). See
// `docs/_meta/rt_measurement.md` for the full design.
//
// Mirrors the `GL_DISINT_PROFILE` convention used in `compiler.hpp`.
#define RT_MEASUREMENT 0

// Diagnostic-only Phase 1 / Phase 3 internal attribution. Each phase worker
// writes elapsed nanoseconds into its own cache line-free row; the barrier
// folds and prints those rows after the join. The resulting category totals
// are summed worker wall time, deliberately kept separate from the complete
// phase barrier wall. No prover decision reads these measurements. Off by
// default like RT_MEASUREMENT and MEM_MEASUREMENT.
#define PHASE13_DEEP_TIMING 0

// When 1, every logical block's end-of-burst size is polled per statified
// container and folded into one process-wide table, the memory counterpart of
// the RT section table. Each active LB is sampled at the end of
// `performElemPhase3` (it is claimed and resident there — a deloaded container
// asserts on `size()`, so an idle LB is unpollable by construction), the
// per-worker rows are folded at the end-of-iteration barrier together with the
// four pool footprints, the scratch registries and the derived indexes, and the
// whole vector is snapshotted whenever its grand total sets a new high-water.
// `main.cpp` writes the peak snapshot once at end of batch to
// `.rt/_memory_<tag>.log`, with each structure's share of that peak. Pure
// telemetry: nothing in the prover reads it, and no deload or steward decision
// depends on it (I-106 / Rule 16). When 0, every call site compiles to nothing.
// See `docs/agentic_swdd/_meta/memory_measurement.md`.
#define MEM_MEASUREMENT 0


#include <cstdint>

// Subset of parameters needed for quick mode scaffold.
// Values mirror GL/parameters.py (quick defaults).
namespace gl {

    /// @brief The prover's per-LB `NameMap` id type — a signed 32-bit integer.
    ///
    /// @details
    /// Every NameMap id (statement originals, validity-scope names, argument
    /// names, the packed-key halves, the validity-forest parent/own-sub ids, the
    /// `IntEncodedExpr` / `IntNormalizedKey` / `NormKey` / `EqClassKey` id fields,
    /// the owner-set partition/signature ids) is a `NameId`. It was `int16_t`
    /// until the id space outgrew the ~32k int16 ceiling (FTA rung-2 mints
    /// >32679 ids); widening to a signed 32-bit alias raises the wall to ~2.1e9
    /// while keeping the ids signed (a `-1` never reaches a NameMap id — the
    /// sentinel-bearing `IntEncodedExpr` fields such as `maxIteration` /
    /// `argIteration` stay a plain `int32_t`, deliberately NOT `NameId`, so the
    /// two roles never blur). The alias is the single knob: retyping it flips
    /// every id field, packer, codec, and stack buffer in lockstep, and the
    /// compiler rejects any narrowing seam a partial migration would leave.
    ///
    /// @invariant `NameId` is a signed, at-least-32-bit integer; the id ceiling
    ///            `ExecutionParameters::MAX_NAME_IDS` is a `NameId` and every mint
    ///            site asserts `id < MAX_NAME_IDS`.
    /// @see `ExecutionParameters::MAX_NAME_IDS`, `NameMap`, `IntEncodedExpr`.
    using NameId = std::int32_t;

    struct ProverParameters {
        // Defaults from parameters.py / old parameters.hpp
        int sizeAllBinariesAna = 10;
        int maxIterationNumberProof = 30;
        int numberIterationsConjectureFiltering = 1;
        int maxSizeDefSetMapping = 5;
        int maxSizeTargetSetMapping = 12;
        int maxNumberSecondaryVariables = 2;
        // Scoped widening of the distinct-secondary cap: applies ONLY to a
        // request whose premises ALL sit at ONE shared validity scope that is
        // an _orint_ branch. Default equal to the standard cap = no widening;
        // a batch config may raise it (rung-2 needs 3 for the second
        // predecessor level's successor-addition firing inside the _orint_
        // branch). Consumed by requestGatesPass.
        int maxNumberSecondaryVariablesOrint = 2;
        int sizeAllPermutationsAna = 7;
        int minNumOperatorsKey = 2;
        int minNumOperatorsKeyCE = 4;
        int maxIterationNumberVariable = 1;
        int standardMaxSecondaryNumber = 1;

        // --- Submatch cap (LB split) ---
        // Maximum number of SUBMATCHES a single LB part may make in one hashburst.
        // The burst stops (BurstSink::canAccept, which the request generator's
        // grow-DFS also consults) once a part reaches this, and
        // the busiest part's submatch count vs this cap drives the split policy.
        // Lowering it is only sound together with LB splitting (each part stays
        // under the cap). See D-109.
        int maxNumberHashRequests = 40000;

        // --- Second-split submatch cap (stump split) ---
        // The submatch ceiling of a part of an ALREADY rule-split LB
        // (numberOfParts > 1). maxNumberHashRequests above stays the ceiling of
        // an UNSPLIT LB's single part, whose cap-hit escalates it to the rule
        // split; a rule-part that reaches THIS lower cap escalates one level
        // further, returning its growing candidates as stumps so the LB splits
        // again on expressions (one sub-part per stump). A stump sub-part that
        // reaches it stops, exactly as a rule-part stops today -- there is no
        // third split. Config key: second_split_submatch_cap. See
        // D-203.
        int second_split_submatch_cap = 10000;

        // --- LB-split disable (diagnostic / RT measurement) ---
        // When true, proveKernel forces one phase-2 part per LB (N = 1) instead
        // of the adaptive numberOfParts, and skips the adaptive escalation /
        // fall-back entirely. With one part per LB, the per-call RTTracker homed
        // in performElem2 measures the LB's whole hashburst -- the only mode in
        // which RT measurement records anything (the split parallel path declares
        // no tracker, so RT is otherwise inert). Byte-identical to the adaptive
        // path for every LB that stays under the cap, but a runaway LB whose
        // single unsplit part exceeds maxNumberHashRequests truncates at the cap
        // with NO re-split, so a disable run is a profiling run, not
        // proof-complete. Default false leaves the production parallel path
        // untouched. See D-110,
        // D-111.
        bool disable_lb_split = false;

        // --- LB-split master switch (config-tunable) ---
        // When false, proveKernel excludes every LB of this batch from the
        // statistics-driven split: the end-of-iteration straggler pass never
        // raises numberOfParts and no producer task is dispatched, so each LB
        // runs one phase-2 part. The flag replaces the former hard incubator
        // exclusion (the gate no longer consults incubator_mode): a batch opts
        // out per config instead. Every production incubator config enables
        // the split so a heavy LB can use the same statistics-driven straggler
        // policy as a main batch; the trigger leaves balanced small LBs
        // unsplit. Distinct from disable_lb_split, which stays the diagnostic /
        // RT-profiling switch. Config key: lb_split.
        bool lb_split = true;

        // --- LB-split growth factor (UNUSED) ---
        // Multiplier of the retired graduated split policy (the `>= 50% fill ->
        // x growthFactor` band of the removed computeNextNumberOfParts). The
        // adaptive policy is now bang-bang 1 <-> fixed_number_splits
        // (adaptiveSplitDecision), so this knob no longer affects the part count.
        // Kept as a no-op for config-file compatibility. See
        // D-111.
        int split_growth_factor = 2;

        // --- LB-split max parts (UNUSED) ---
        // Upper bound of the retired graduated split policy (the cap of the
        // removed computeNextNumberOfParts). The adaptive policy is now bang-bang
        // 1 <-> fixed_number_splits (adaptiveSplitDecision), which never exceeds
        // fixed_number_splits, so this separate bound no longer applies. Kept as
        // a no-op for config-file compatibility. See
        // D-111.
        int max_number_splits = 32;

        // --- LB-split ESCALATION TARGET (config-tunable) ---
        // The part count an LB jumps to when its unsplit (numberOfParts == 1)
        // burst HITS the submatch cap: proveKernel discards that truncated burst
        // and re-runs the LB from scratch at this many parts in the SAME
        // iteration, and the LB stays here on later iterations until it falls
        // back (split_fallback_ratio). Every main-path / compressor LB starts
        // unsplit (numberOfParts default 1) and escalates only on demand. A
        // batch whose config sets lb_split false runs UNSPLIT (the incubator
        // configs with thousands of small LBs, whose per-part request-gen setup
        // is redundant). The CE filter is unaffected (its
        // own un-split loop, splitCount=1, never reaches proveKernel).
        // Raise/lower to trade split overhead against per-part submatch load. See
        // D-111.
        //
        // It is ALSO the stump-count floor produceExpressionStumps grows a short
        // filter list toward. A rule-part that reaches second_split_submatch_cap
        // returns one stump per expression surviving its request filter, and one
        // sub-part runs per stump, so the two dimensions multiply. At 100 rule-
        // parts a single Peano induction LB drew 13,188 sub-parts and its hash
        // burst took 108 s, dominated by the fixed per-part cost (the statement
        // filter and the obligatory-stump builders, five batches over, rebuilt by
        // every part). 20 keeps the product in a range where the mechanism can be
        // exercised end to end; a filter list above 20 expressions never grows,
        // so a stump stays one expression.
        int fixed_number_splits = 20;

        // --- LB-split FALL-BACK ratio (config-tunable) ---
        // A split LB (numberOfParts > 1) coarsens back to unsplit (numberOfParts
        // = 1) for its NEXT burst when its BUSIEST part's submatch count this
        // burst is below this fraction of maxNumberHashRequests -- i.e. every
        // part ran well under the cap, so the split is buying nothing. The
        // escalation direction is NOT ratio-gated: an unsplit LB jumps straight
        // to fixed_number_splits the moment its single part hits the cap (see
        // adaptiveSplitDecision). Default 0.10. Tuning note: a value above
        // 1/fixed_number_splits lets a medium-sized LB (total work between 1x and
        // ~1/ratio x the cap) oscillate 1 <-> fixed_number_splits each iteration
        // -- a capped unsplit pass wasted per oscillation, but the proof output
        // is unchanged (every APPLIED burst is complete). Config key:
        // split_fallback_ratio. See D-111.
        double split_fallback_ratio = 0.10;

        // --- LB-split setup break-even floor (config-tunable) ---
        // The straggler trigger's floor (isStraggler): an LB splits into
        // logicalCores expression buckets next iteration only if its total submatch
        // work this iteration exceeds BOTH the idle-core fair-share (T / logicalCores)
        // AND this floor. It is the per-bucket setup break-even -- each of the
        // logicalCores buckets re-pays the fixed setup (filterIntEncodedStatements +
        // the obligatory-stump builder over the whole statement universe), so below
        // the point where a bucket's share of the work exceeds that setup, splitting
        // cannot pay off. This is the ONLY tunable knob of the split trigger (the
        // fan-out is logicalCores, not a config number).
        //
        // The submatch tally it is compared against counts grow-search nodes only:
        // the pairing merge that used to add one count per (base candidate, stump)
        // attempt is gone with the mandatory-containment control, so the same
        // proof reaches a far smaller total than it did under the merge.
        // Config key: min_split_work.
        int min_split_work = 5000;

        // NOTE: maxNumberHashRequests / second_split_submatch_cap /
        // fixed_number_splits / split_fallback_ratio above are UNUSED on the main
        // path since the split trigger became statistics-driven and preemptive: a
        // straggler is split into logicalCores expression buckets next iteration
        // (isStraggler), bursts run to completion (no cap, no escalation). Kept as
        // no-op fields only to keep config-file parsing stable.

        // --- Quiescent-burst skip (D-194) ---
        // When true, proveKernel's single-threaded active-build excludes from the
        // sweep every active LB whose burst would provably do nothing — no
        // producer-side work (hasWork clear) and no un-ingested ancestor mail
        // (mailPeek false). A skipped LB stays isActive and is never paged in,
        // collapsing the 4 GiB paging of the converged frozen tail. Warm-up
        // iterations and compressor mode never skip. Set false to A/B against the
        // sweep-everything reference (must be byte-identical). Config key:
        // enable_quiesce_skip. See D-194, I-153.
        bool enable_quiesce_skip = true;

        // --- Extent-file raw eviction (D-195) ---
        // When true, the v4 raw eviction images live in ONE preallocated extent
        // file, each LB placed at a stable slab offset and dumped/reloaded by
        // positioned I/O into the already-open handle — eliminating the per-
        // eviction NTFS create/truncate/close (and the per-file Defender scan).
        // When false, the raw path falls back to one named file per LB (the
        // pre-extent behaviour) for A/B comparison. Default true on the branch.
        // Config key: enable_extent_deload.
        bool enable_extent_deload = true;

        // Per-batch Phase 2 execution policy. The processor route is the
        // default for every run (a customer PC needs no GPU); `main.py --GPU`
        // passes `--phase2-backend cuda` to every batch, and a config may pin
        // one batch to CUDA by setting this true. A selected CUDA route that
        // finds its device, driver or runtime missing asserts — no fallback.
        bool use_gpu = false;

        // Per-batch permission for the working-set steward to write LB images
        // to SSD. Backend-derived at prover construction: a processor batch
        // keeps this default and pages; CUDA and SSD deload are intentionally
        // mutually exclusive, so a CUDA batch derives false (resident-only
        // steward). A config may still set false to pin a processor batch
        // resident. Config key: allow_ssd_deload.
        bool allow_ssd_deload = true;

        bool trackHistory = true;
        int inductionMaxAdmissionDepth = 1;
        int inductionMaxSecondaryNumber = 2;
        int counterExampleBoundary = 6;
        int minLenLongKey = 5;
        int maxLenHypoKey = 2;
        bool debug = false;

        // --- Compressor Parameters ---
        bool compressor_mode = false;
        bool ban_disintegration = false;
        int max_origin_per_expr = 1;                 // cap used during normal prover runs
        int compressor_max_origins_per_expr = 30;    // cap used during compressor Phase 1 hash bursts
        int compressor_hash_bursts = 15;

        // --- Incubator Parameters ---
        bool try_contradiction = false;
        // Complement of try_contradiction: every registered conjecture also
        // gets an LB that ASSUMES the negation of its head and, on
        // contradiction, emits the conjecture itself as proved (reductio).
        bool try_contradiction_negated_head = false;
        bool skip_ce_filter = false;
        // Mirror-refutation heuristic (D-229):
        // a CE-refuted operator-only conjecture also refutes its pool mirror
        // via mirror_pairs.txt. Gates ONLY the CE-filter flip pass — never a
        // proof step (I-81 / D-112 untouched).
        bool mirror_refutation = true;
        bool skip_eq_classes = false;
        bool incubator_mode = false;

        // --- Axed-anchor deposit exception ---
        // When true, a POSITIVE anchor-category statement carrying an axed
        // x-copy name passes the axed-variable deposit filter in
        // addExprToMemoryBlock (D-234), so an
        // anchor-bridge rule firing on the x-copied batch anchor lands the
        // external anchor's x-form as a live statement. Default false: the
        // x-form deposit stays filtered — a live x-anchor opens the whole
        // external anchor's rule universe at x-arguments, which explodes
        // the batch runtime. Config key: axed_anchor_exception.
        bool axed_anchor_exception = false;

        // --- multiplyImplication gate (decoupled from incubator_mode) ---
        // allow_multiplication gates multiplyImplication at prover.cpp:827.
        // Default false = main-path behavior (multiplication off in plain
        // prove pass; CE-filter still multiplies regardless of this flag).
        // Pass B gating is on ban_disintegration above (negative-sense, !ban
        // = Pass B fires); the short-lived allow_disintegration flag was
        // collapsed into ban_disintegration on 2026-04-29 — see D-28.
        bool allow_multiplication = false;

        // --- multiplyImplication ---
        int max_partition_size = 5;

        // --- OR expression handling ---
        int max_or_depth = 1;   // max nesting depth of OR scopes (1 = no nested ORs)

        // --- Statification: static per-LB memory sizing ---
        // The static-memory machinery makes ONE reservation of
        // static_pool_bytes at program start (never freed, never grown) and
        // carves it into equal blocks of static_block_bytes — the grant unit
        // dispensed to each LB's manager — which carves each granted block
        // into equal pages of static_page_bytes, the fixed unit the paged
        // containers allocate and return. These three sizes — and the
        // persistent-pool and hot-arena sizes below — are the SOLE source of
        // truth: fixed here, identical for every batch, never read from any
        // config (D-139). The sizing must satisfy
        // isValidStaticMemoryConfig (pool a whole multiple of block, block a
        // power of two for the arena's offset shift/mask) and
        // isValidStaticPageConfig (block a whole multiple of page, page a
        // power of two); the ExpressionAnalyzer constructor asserts both.
        int64_t static_pool_bytes = 12884901888LL; // 12 GiB
        int32_t static_block_bytes = 262144;      // 256 KiB
        int32_t static_page_bytes = 8192;         // 8 KiB

        // --- Statification: persistent (second) pool ---
        // A SECOND program-start reservation, separate from static_pool_bytes
        // and NEVER deloaded. It backs Memory::intToBeProved so the deactivation
        // survey can read the goal registry regardless of the main arena's
        // deload state (the determinism fix). Smaller blocks than the main pool
        // (the per-LB persistent content is tiny); shares static_page_bytes.
        // Each active LB pins >=1 persistent block for its whole active life
        // (reclaimed only at discharge), so size by peak active-LB count, not
        // content. Must satisfy isValidStaticMemoryConfig (pool/block, block a
        // power of two) and isValidStaticPageConfig (block/page); the
        // ExpressionAnalyzer constructor asserts both. Exhaustion asserts naming
        // static_persistent_pool_bytes.
        int64_t static_persistent_pool_bytes = 1073741824LL; // 1 GiB
        int32_t static_persistent_block_bytes = 32768;       // 32 KiB (4 pages)

        // --- Statification: mail (third) pool ---
        // A THIRD program-start reservation, separate from both pools above and
        // NEVER deloaded. It backs the cross-LB pull-model mail system: the
        // ExpressionAnalyzer MailLog plus per-LB incoming routing mailboxes and
        // the global mail interner. MailLog keeps full history for a grid with an
        // initially dormant LB and rolling delivered windows otherwise (I-161).
        // A stand-alone pool by design — nothing reads its grant ledger,
        // so it plays no role in any deload/throttle/steward decision and the
        // mail content never competes with the deloadable main pool. Shares
        // static_page_bytes; blocks match the main pool's (mail content is
        // bulky, so large blocks keep grant traffic low). Size by peak total
        // mail-system content across a batch (tune by `[pool-memory]`); exhaustion
        // asserts naming static_mail_pool_bytes. Must satisfy
        // isValidStaticMemoryConfig (pool/block) and isValidStaticPageConfig
        // (block/page); the ExpressionAnalyzer constructor asserts both.
        int64_t static_mail_pool_bytes = 2147483648LL;       // 2 GiB
        int32_t static_mail_block_bytes = 262144;            // 256 KiB

        // --- Statification: LB-body (fourth) pool ---
        // A FOURTH program-start reservation, separate from all three pools above
        // and NEVER deloaded. It backs the LB object store (LbStore on
        // ExpressionAnalyzer): the Memory node objects themselves are placement-
        // new'd into fixed-size slots carved from this pool's blocks instead of
        // the malloc heap. A stand-alone pool by design — nothing reads its grant
        // ledger, so it plays no role in any deload/throttle/steward decision, and
        // the shells never compete with the deloadable main pool. Shares
        // static_page_bytes (the store carves blocks into Memory-sized slots, not
        // pages, but the config must still satisfy the page contract); blocks must
        // exceed sizeof(Memory) (one slot never straddles a block — asserted at
        // first allocation). Size by peak simultaneously-live LB count x
        // sizeof(Memory) (tune by the pool telemetry); exhaustion asserts naming
        // static_lb_pool_bytes. Must satisfy isValidStaticMemoryConfig (pool/block,
        // block a power of two) and isValidStaticPageConfig (block/page); the
        // ExpressionAnalyzer constructor asserts both.
        int64_t static_lb_pool_bytes = 2147483648LL;         // 2 GiB
        int32_t static_lb_block_bytes = 262144;              // 256 KiB
    };

    /// @brief Pure validity predicate for the statification memory-sizing
    ///        pair (`static_pool_bytes`, `static_block_bytes`).
    ///
    /// @details
    /// The static-memory machinery carves one program-start reservation of
    /// `poolBytes` into equal blocks of `blockBytes`, and each LB's bump arena
    /// bump-allocates within its blocks. The pool carves into whole blocks
    /// only when `poolBytes % blockBytes == 0`, and the arena resolves an
    /// offset to a physical address with a shift/mask that requires
    /// `blockBytes` to be a power of two; both with strictly positive values.
    ///
    /// Deliberately a pure predicate rather than an asserting routine: call
    /// sites assert on its result (a bad pair is a misconfiguration that must
    /// stop the run at config load, not at first allocation — I-19), while
    /// unit tests exercise the rejecting branches directly without aborting
    /// the harness.
    ///
    /// @param poolBytes  Total bytes of the one program-start reservation.
    /// @param blockBytes Bytes per block, the grant unit handed to each LB's
    ///                   arena; must be a power of two.
    /// @return `true` when both values are strictly positive, the pool divides
    ///         into whole blocks, and the block size is a power of two;
    ///         `false` otherwise.
    /// @invariant Pure function of its arguments — no state, no side effects.
    /// @see `ProverParameters::static_pool_bytes` and sibling for the config
    ///      fields this validates.
    [[nodiscard]] constexpr bool isValidStaticMemoryConfig(
        int64_t poolBytes, int32_t blockBytes) noexcept
    {
        return poolBytes > 0 && blockBytes > 0
            && poolBytes % blockBytes == 0
            && (blockBytes & (blockBytes - 1)) == 0;
    }

    /// @brief Pure validity predicate for the page tier of the statification
    ///        memory hierarchy (`static_block_bytes`, `static_page_bytes`).
    ///
    /// @details
    /// Each LB manager carves a granted block into equal pages of `pageBytes`;
    /// the carve produces no partial page only when `blockBytes % pageBytes ==
    /// 0`, and a power-of-two `pageBytes` keeps the paged container's
    /// within-page index a shift/mask. Both values strictly positive.
    ///
    /// Kept separate from `isValidStaticMemoryConfig` (the pool/block pair) so
    /// that predicate's existing two-argument call sites stay unchanged; the
    /// ExpressionAnalyzer constructor asserts both. Deliberately a pure predicate asserted
    /// at the call site (a bad page size is a misconfiguration that must stop
    /// the run at config load, not at first allocation — I-19), so unit tests
    /// exercise the rejecting branches without aborting the harness.
    ///
    /// @param blockBytes Bytes per block, the grant unit; must be a whole
    ///                   multiple of `pageBytes`.
    /// @param pageBytes  Bytes per page, the per-LB allocation unit; must be a
    ///                   power of two and divide `blockBytes`.
    /// @return `true` when both values are strictly positive, the block divides
    ///         into whole pages, and the page size is a power of two; `false`
    ///         otherwise.
    /// @invariant Pure function of its arguments — no state, no side effects.
    /// @see `ProverParameters::static_page_bytes`, `isValidStaticMemoryConfig`.
    [[nodiscard]] constexpr bool isValidStaticPageConfig(
        int32_t blockBytes, int32_t pageBytes) noexcept
    {
        return blockBytes > 0 && pageBytes > 0
            && blockBytes % pageBytes == 0
            && (pageBytes & (pageBytes - 1)) == 0;
    }

    // Static hot path sizing constants — config-independent, compile-time.
    struct ExecutionParameters {
        static constexpr int16_t MAX_KEY_SLOTS   = 256;    // max NameId values in a normalized key
        static constexpr NameId  MAX_NAME_IDS    = 1000000; // max NameMap IDs (raised from 32000 now that NameId is 32-bit; mint-time tripwire at every mint site)
        static constexpr int32_t KEY_ARENA_CHUNK = 16384;  // NameId slots per arena chunk (unused knob)
        static constexpr int16_t MAX_EXPRESSIONS = 8;      // max expressions in a single key
        static constexpr int16_t MAX_ARITY       = 16;     // max arguments per expression
        static constexpr int32_t MAX_SCOPE_DEPTH = 64;     // max validity ancestor-chain length incl. self; assert tripwire, see strictAncestorSpans

        // Max bytes of an eqClassSttmntIndexMapMap packed key: a NameId LE
        // validity id then up to MAX_ARITY*MAX_KEY_SLOTS member ids (the
        // reduceEqClassIds class-member ceiling), each sizeof(NameId) bytes.
        // Sizes the stack key buffers in encodeEqClassKeyFromViewInto /
        // …AccumInto (loud widen-on-STOP assert = Rule-19 tripwire).
        static constexpr int32_t kMaxEqClassKeyBytes =
            static_cast<int32_t>(sizeof(NameId))
            * (static_cast<int32_t>(MAX_ARITY)
                 * static_cast<int32_t>(MAX_KEY_SLOTS) + 1);

        // Max bytes of a Codec<NormKey> key/record: NameId-width numberExpressions
        // + NameId-width length + up to MAX_KEY_SLOTS NameId data (the record is
        // uniform 4-byte). Sizes the stack key buffer in encodeNormKeyInto (loud
        // widen-on-STOP assert = Rule-19).
        static constexpr int32_t kMaxNormKeyBytes =
            static_cast<int32_t>(sizeof(NameId))
            * (static_cast<int32_t>(MAX_KEY_SLOTS) + 2);


        /// @brief Chunk size of `applyFiringRecords`' pointer-index sort — the
        ///        longest contiguous `int32_t` index one arena block holds.
        ///
        /// @details
        /// The sort index rides the byte-bump tier, where a single allocation
        /// must fit ONE pool block (`ProverParameters::static_block_bytes`,
        /// default 256 KiB): this constant is that default divided by
        /// `sizeof(int32_t)`, 65,536 slots. A burst producing more firings than
        /// that cannot be sorted in one contiguous run, so the index is cut into
        /// chunks of this size, each `std::sort`ed on its own, and the chunks are
        /// consumed merged. At or below this many records there is exactly ONE
        /// chunk, which is a single `std::sort` over one contiguous index — the
        /// form this path had before the chunking, at the same cost.
        ///
        /// It was formerly a hard ceiling (`kMaxFiringRecordsPerLbBurst`) with an
        /// assert. The stump split pushed a Peano induction LB past it: its
        /// rule-parts stopped truncating at the submatch cap, so the burst ran to
        /// completion and produced 80,147 records. The ceiling was a property of
        /// the allocation, never of the merge, so it became a chunk size rather
        /// than being raised. Anyone retuning `static_block_bytes` must retune
        /// this in step.
        ///
        /// @see `applyFiringRecords` — the original user;
        ///      `ChunkSortedOrdinals` — the reusable generalization (the
        ///      default chunk size of the mail-absorb row indexes).
        static constexpr int32_t kFiringRecordSortChunk = 262144 / 4;

        /// @brief Ceiling on the number of sort chunks one LB burst may need.
        ///
        /// @details
        /// Sizes `applyFiringRecords`' stack arrays of per-chunk cursors, so it
        /// bounds a burst at `kFiringRecordSortChunk * this` = 4,194,304 firing
        /// records. Two orders of magnitude above the largest burst observed
        /// (80,147). Overrun is a loud assert (Rule 19), never a truncated sort.
        ///
        /// @see `applyFiringRecords` — the guarded consumer.
        static constexpr int32_t kMaxFiringRecordSortChunks = 64;

        /// @brief Ceiling on one admission / rejected key's serialized RUN
        ///        byte size — sizes the whole-run concatenation buffer in
        ///        `insertAdmissionBlobSorted` / `insertRejectedBlobSorted`.
        ///
        /// @details
        /// The D-172 RMW splice concatenates a key's whole canonical run
        /// (survivor blobs + the new blob) into ONE contiguous byte-bump
        /// allocation before the raw `assignRun` write-back; a single
        /// byte-bump allocation must fit one pool block
        /// (`ProverParameters::static_block_bytes`, default 256 KiB), and
        /// this constant is that default. The retired heap path (a
        /// `std::vector<char>` assembled by the typed `assignRun`) had no
        /// such ceiling; current corpora sit far below it (admission /
        /// rejected runs are tens of records of tens of bytes). The named
        /// assert at the two splice sites makes a scale overrun read as the
        /// designed capacity bound — never as corruption via the arena's
        /// generic assert, and never a silent clamp (Rule 19). If it ever
        /// fires, the documented widening path is chunked assembly through
        /// the engine's `appendBlobToRun` (per-blob appends instead of one
        /// contiguous concat). Anyone retuning `static_block_bytes` must
        /// retune this constant in step.
        ///
        /// @see `insertAdmissionBlobSorted`, `insertRejectedBlobSorted` —
        ///      the guarded concatenation sites.
        static constexpr int32_t kMaxAdmissionRunBytes = 262144;

        /// @brief Ceiling on one origin history line's dependency count —
        ///        sizes the per-blob stack buffers of the origin RMWs
        ///        (`addMailOriginRecord`, the cold `addOriginId`) and their
        ///        serializers (`serializeMailOriginTo`, `serializeOriginTo`).
        ///
        /// @details
        /// An origin record's dependencies come from caller stack
        /// `OriginDep[]` runs whose realistic counts are single-digit (an
        /// antecedent list), so 64 is generous headroom. The named asserts at
        /// the serializer fill sites are Rule-19 tripwires: an overrun reads
        /// as the designed capacity bound, never a silent clamp; the
        /// documented widening path is raising this constant with evidence.
        ///
        /// @see `serializeMailOriginTo`, `serializeOriginTo`,
        ///      `addMailOriginRecord`, `addOriginId` — the guarded sites.
        static constexpr int32_t kMaxOriginDeps = 64;

        /// @brief Byte size of one origin history line's serialized blob at
        ///        the `kMaxOriginDeps` ceiling — the per-blob stack buffer
        ///        size of the origin RMWs.
        ///
        /// @details
        /// The `Codec<IntMailOrigin>` / `Codec<IdOrigin>` closed-form frame:
        /// `uint8 tag` + `int32 depCount` + `depCount x int64 dep` =
        /// `5 + 8 * depCount` bytes.
        ///
        /// @see `kMaxOriginDeps`, `serializeMailOriginTo`, `serializeOriginTo`.
        static constexpr int32_t kMaxOriginBlobBytes = 5 + 8 * kMaxOriginDeps;

        /// @brief Ceiling on one origin key's RUN length (blob count) at the
        ///        cap-full rebuild — sizes the whole-run stack buffer of the
        ///        origin RMWs' D-49 convenience-replace branch.
        ///
        /// @details
        /// Origin runs are config-capped (`max_origin_per_expr`, default 1;
        /// `compressor_max_origins_per_expr`, 30), so 40 is headroom above
        /// every production cap. The rebuild branch asserts the ACTUAL run
        /// length against this constant (a run written under a larger past
        /// cap, or an uncapped mail fold reaching a finite-cap rebuild, would
        /// fire it loudly — Rule 19, widen with evidence, never soften).
        ///
        /// @see `addMailOriginRecord`, `addOriginId` — the rebuild sites.
        static constexpr int32_t kMaxOriginRunBlobs = 40;

        /// @brief Byte size of the origin RMWs' whole-run rebuild stack
        ///        buffer — `kMaxOriginRunBlobs` blobs at `kMaxOriginBlobBytes`
        ///        each.
        ///
        /// @see `kMaxOriginRunBlobs`, `kMaxOriginBlobBytes`.
        static constexpr int32_t kMaxOriginRunBytes =
            kMaxOriginRunBlobs * kMaxOriginBlobBytes;

        /// @brief Ceiling on the element count of one admission-map KEY chain
        ///        — sizes the fixed `StrSpan` / `SealedString` run buffers the
        ///        admission gates carry a key through (`renamingChainScratch`,
        ///        `updateAdmissionMap`, the deferred-admission drain,
        ///        `isAdmitted`).
        ///
        /// @details
        /// An admission key is a normalized-key chain of expressions, bounded
        /// in production by `MAX_EXPRESSIONS` (8) — this constant is an 8x
        /// headroom bound so the fixed stack runs never clamp a legitimate
        /// key. The named asserts at each run's fill site make a scale overrun
        /// read as the designed capacity bound it is (Rule 19), never a silent
        /// truncation; if one ever fires the documented widening path is a
        /// page-tier `PagedVector` run in place of the contiguous stack array.
        ///
        /// @see `renamingChainScratch`, `updateAdmissionMap`,
        ///      `drainDeferredAncestorAdmissions`, `isAdmitted`.
        static constexpr int32_t MAX_ADMISSION_KEY_ELEMENTS = 64;

        /// @brief Ceiling on the count of an admission key's remaining-argument
        ///        run (the sorted-unique `u_`-stripped argument set) and of
        ///        `renamingChainScratch`'s distinct-argument working sets.
        ///
        /// @details
        /// The remaining-args set is a subset of the distinct `u_` arguments
        /// present across the key chain; the absolute ceiling on distinct
        /// arguments in a normalized key is `MAX_EXPRESSIONS * MAX_ARITY`
        /// (128), and this constant sits above it so neither the remaining-args
        /// run nor the renaming pass ever clamps. The named asserts at each
        /// fill site are Rule-19 tripwires (widen with evidence, never soften).
        ///
        /// @see `renamingChainScratch`, `updateAdmissionMap`,
        ///      `drainDeferredAncestorAdmissions`, `isAdmitted`.
        static constexpr int32_t MAX_ADMISSION_REM_ARGS = 256;

        /// @brief Minimum operator-expression count (head included, anchors
        ///        excluded) for the ordis-only admission qualification route.
        ///
        /// @details
        /// A rule failing the regular (A)/(B)/(C) admission gates still
        /// installs `ordisOnly`-tagged marker keys iff every non-anchor
        /// element and the head are operator applications and their count
        /// reaches this bound. 4 is the maintainer-confirmed boundary: the
        /// A15 carrier (corpus row 34, 3 premises + head) passes exactly at
        /// it. The documented tightening knob if the route proves too loose
        /// is the head-gate of the design's decision 8, not this number.
        ///
        /// @see `ExpressionAnalyzer::ordisRouteQualifies` — the consumer.
        static constexpr int32_t kOrdisMinOperatorExpressions = 4;

        /// @brief Minimum argument count for an expression to enter the
        ///        ordis2 demand/park key language
        ///        (D-267).
        ///
        /// @details
        /// Filters equalities (2 args) and `in2`-style membership facts
        /// (3 args) out of both halves of the pair — demand for equalities
        /// would be minted by half the pool and sweep constantly. The
        /// order compacts `preorder` / `strictOrder` (4 args) pass. Shared
        /// by the demand slot filter and the park-side disjunct filing so
        /// the two key populations can never diverge.
        ///
        /// @see `ExpressionAnalyzer::ordis2KeyEligible` — the one consumer.
        static constexpr int32_t kOrdis2DemandMinArity = 4;

        /// @brief Minimum NON-anchor premise count (qualifying slot
        ///        included) for a rule to install ordis2 demand variants
        ///        (D-267, maintainer-set
        ///        2026-08-10).
        ///
        /// @details
        /// Without it, every rule with any qualifying ≥4-arg compound slot
        /// installs demand-marker families across all LBs — the first
        /// post-fix acceptance run's RT explosion. 4 admits exactly the
        /// premise-rich consumer shape B8 needs (B5: strictOrder slot +
        /// preorder guard + two products) and excludes the short rules
        /// whose demand traffic is pure fan-out.
        ///
        /// @see `ExpressionAnalyzer::ordis2DemandSlotQualifies` — the one
        ///      consumer.
        static constexpr int32_t kOrdis2DemandMinPremises = 4;

        /// @brief Minimum NON-anchor premise count for a rule to install
        ///        UNTAGGED input-slot demand variants
        ///        (D-288).
        ///
        /// @details
        /// The algebra twin of `kOrdis2DemandMinPremises`, guarding the
        /// input-slot demand pass against the same fan-out the ordis2 gate
        /// was set for: without it every short rule with a head-linked
        /// confined input slot installs marker families across all LBs. 4
        /// admits the premise-rich witness-consumer shape C8 needs (the
        /// difference-transport rule: three product/sum premises binding
        /// the slot's other args, the slot itself carrying the witness)
        /// and excludes the short introduction rules.
        ///
        /// @see `ExpressionAnalyzer::inputSlotDemandSlotQualifies` — the one
        ///      consumer.
        static constexpr int32_t kInputSlotDemandMinPremises = 4;

        /// @brief Ceiling on one registered or-operator's FLATTENED leaf
        ///        count for the reduced-or pre-mint closure.
        ///
        /// @details
        /// `preMintReducedOrs` registers the single-elimination closure of
        /// every registered pool or (k reduced (k-1)-ary operators per
        /// k-ary or, recursively). The closure size grows combinatorially
        /// in k, so a runaway leaf count must fail loudly at the pre-mint
        /// seam, not silently flood the registry. The current pool tops
        /// out at k = 3 (trichotomy); 8 leaves headroom above that. A
        /// firing assert names this constant — widen with evidence, never
        /// soften (Rule 19).
        ///
        /// @see `ExpressionAnalyzer::preMintReducedOrs` — the consumer.
        static constexpr int32_t kMaxReducedOrLeaves = 8;

        /// @brief Rule-19 tripwire on the subset-exclusion detector's
        ///        per-registry-entry embed search.
        ///
        /// @details
        /// `isSubsetExclusionInstall` matches one registry or's flattened
        /// leaf list against the head's leaves plus the premise negations
        /// by a backtracking embed walk. The true worst case at
        /// `kMaxReducedOrLeaves = 8` is orders of magnitude below this
        /// cap; a firing assert means a contract break (runaway
        /// unification, an oversized flat entity), never a tuning knob.
        ///
        /// @see `ExpressionAnalyzer::isSubsetExclusionInstall` — the consumer.
        static constexpr int32_t kSubsetExclusionEmbedCap = 4096;

        /// @brief Ceiling on the ENTRY count of one admission-integration key's
        ///        nested instruction map — sizes the caller-owned `int32_t[]`
        ///        run that `ArenaIntegrationMap::sortedIndices(out, cap)` fills.
        ///
        /// @details
        /// One admission-integration key (`packStatementKey(templateId,
        /// validityId)`) maps to a nested `ArenaIntegrationMap` whose ENTRIES are
        /// the distinct instruction templates admitted under it. At current
        /// corpus scale an entry run is tens of records (the cold blob run is
        /// "tens of records of tens of bytes"); this constant sits two orders of
        /// magnitude above it so the caller-fill snapshot never clamps a
        /// legitimate key. The retired heap `sortedIndices()` `std::vector<
        /// int32_t>` had no such ceiling. The named asserts at each fill /
        /// consume site (`ArenaIntegrationMap::sortedIndices`, and the four
        /// external snapshot sites `isAdmittedIntegration` /
        /// `updateAdmissionMapIntegration` /
        /// `applyEquivalenceClassToAdmissionMapIntegration` / `writeToCold`) make
        /// a scale overrun read as the designed capacity bound it is (Rule 19),
        /// never a silent clamp or truncated sort. If it ever fires, the
        /// documented widening path is a page-tier `PagedVector<int32_t>` index
        /// sorted through paged storage instead of one contiguous stack run.
        ///
        /// @see `ArenaIntegrationMap::sortedIndices`, `isAdmittedIntegration`,
        ///      `updateAdmissionMapIntegration`,
        ///      `applyEquivalenceClassToAdmissionMapIntegration`.
        static constexpr int32_t MAX_INTEGRATION_ENTRIES = 4096;

        /// @brief Ceiling on the VALUE count of one admission-integration entry's
        ///        value set — sizes the caller-owned `int32_t[]` run that
        ///        `ArenaIntegrationMap::valuesAt(ei, out, cap)` fills.
        ///
        /// @details
        /// Each `ArenaIntegrationMap` entry carries a `DecodedIdLess`-sorted set
        /// of value ids (the arguments recorded against the instruction). Like
        /// the entry count it is tens of ids in practice; this constant sits far
        /// above it so the caller-fill value run never clamps. The retired heap
        /// `valuesAt()` `std::vector<int32_t>` had no ceiling. The named assert at
        /// the fill site and the consume site
        /// (`applyEquivalenceClassToAdmissionMapIntegration`) is a Rule-19
        /// tripwire; the widening path is a page-tier `PagedVector<int32_t>` run
        /// in place of the contiguous stack array. Widen only with evidence.
        ///
        /// @see `ArenaIntegrationMap::valuesAt`,
        ///      `applyEquivalenceClassToAdmissionMapIntegration`.
        static constexpr int32_t MAX_INTEGRATION_ENTRY_VALUES = 4096;

        /// @brief Ceiling on one admission-integration key's serialized RUN
        ///        byte size — sizes the whole-run concatenation buffer in
        ///        `ArenaIntegrationMap::writeToCold`.
        ///
        /// @details
        /// `writeToCold` concatenates a key's whole `Codec<IntegrationEntry>`
        /// run (every entry's canonical bytes, in `DecodedInstructionLess`
        /// order) into ONE contiguous byte-bump allocation on the map's own
        /// arena before the raw `assignRun` write-back; a single byte-bump
        /// allocation must fit one pool block
        /// (`ProverParameters::static_block_bytes`, default 256 KiB), and this
        /// constant is that default. The retired heap path (a `std::vector<
        /// char>` grown by the typed `assignRun`) had no such ceiling; current
        /// corpora sit far below it (an integration run is tens of records of
        /// tens of bytes). The named assert at the concatenation site makes a
        /// scale overrun read as the designed capacity bound — never as
        /// corruption via the arena's generic fits-one-block assert, and never
        /// a silent clamp (Rule 19). If it ever fires, the documented widening
        /// path is chunked assembly through the engine's `appendBlobToRun`
        /// (per-record appends instead of one contiguous concat). Anyone
        /// retuning `static_block_bytes` must retune this constant in step.
        /// Sibling of `kMaxAdmissionRunBytes` (the admission / rejected RMW
        /// splice's identical whole-run ceiling).
        ///
        /// @see `ArenaIntegrationMap::writeToCold` — the guarded concatenation
        ///      site; `kMaxAdmissionRunBytes` — the sibling precedent.
        static constexpr int32_t kMaxIntegrationRunBytes = 262144;

        /// @brief Ceiling on the ELEMENT count of one integration-template
        ///        logical entity — sizes the fixed `ScratchString[]` /
        ///        `StrSpan[]` element runs the integration cores carry a
        ///        rewritten entity through (`prepareIntegrationCore`'s
        ///        `newElems` recursion run, `prepareIntegrationCore2`'s
        ///        per-entity `leElements` run).
        ///
        /// @details
        /// A compiled integration template's logical entity (`existence` / `and`
        /// / `or`) carries a handful of defining elements; this constant is a
        /// generous headroom bound so the fixed element runs never clamp a
        /// legitimate entity. The retired heap `std::vector<std::string>`
        /// element lists had no ceiling. The named asserts at each fill site are
        /// Rule-19 tripwires; if one ever fires the documented widening path is a
        /// page-tier `PagedVector<ScratchString>` run in place of the contiguous
        /// stack array. Widen only with evidence.
        ///
        /// @see `prepareIntegrationCore`, `prepareIntegrationCore2`.
        static constexpr int32_t MAX_INSTRUCTION_ELEMENTS = 64;

        /// @brief Ceiling on the count of DISTINCT bracketed tokens collected from
        ///        one expression by `collectExprTokens` — sizes the fixed token /
        ///        `StrReplacement` stack runs in `stripUPrefixASTScratch` (and the
        ///        integration builders that reuse it).
        ///
        /// @details
        /// The `stripUPrefixAST` twin walks every `[...]` token of an expanded
        /// implication and keeps the DISTINCT `u_`-prefixed ones for a `u_`-strip
        /// rewrite; the distinct-token count is bounded by the expression's
        /// variable population, which stays well under this generous headroom
        /// bound. The retired heap `std::set<std::string> tokens` had no ceiling.
        /// The named asserts at the fill sites are Rule-19 tripwires; the widening
        /// path is a page-tier `PagedVector<StrSpan>` token run in place of the
        /// contiguous stack array. Widen only with evidence.
        ///
        /// @see `stripUPrefixASTScratch`, `expandSignatureForIntegrationScratch`.
        static constexpr int32_t MAX_EXPR_TOKENS = 512;
    };

    // RT measurement tunables — only consulted when RT_MEASUREMENT == 1.
    // See `docs/_meta/rt_measurement.md` and the `#define RT_MEASUREMENT`
    // block at the top of this file.
    struct RTMeasurementParameters {
        // Online dump fires once a single elementary step has been running
        // this many wall-clock seconds without returning.
        static constexpr int RT_TIME_TRIGGER_SECONDS = 5;

        // Section rows below this share of the call's total elapsed
        // self-time are folded into the trailing "other sections each
        // < N % of total" line instead of appearing as their own row.
        // Set to 0 so every scope (including small parent scopes
        // shrunken by exclusive-time attribution to nested children)
        // is visible — the user wants the full picture, not a filtered
        // top-N view.
        static constexpr int RT_MIN_PERCENTAGE = 0;

        // Maximum number of distinct (label, parent) section rows per call.
        // Stack-only storage; no `new` / `malloc` (the no-heap convention
        // carried by I-95, successor of I-13's removed `ChunkPool`). Sized
        // for the phase-1/3 atomic split: the standardProcessing absorb and
        // the deposit-door tree open one row per (label, parent) pair, and
        // the external/internal absorb parents double the door rows; the
        // hash-memory install scopes (HM_*) open under every install caller
        // (four door targets, the integration prepare, the compact re-expands),
        // which is what pushed the row count past 192; the atomic install
        // dissection (per-permutation key build / per-subkey write / the
        // cold-writer interiors / the CSR store) multiplies those rows by
        // the four hash-memory targets and the admission passes, hence
        // 4096. Heap-backed per tracker since then (`rt_tracker.hpp`).
        static constexpr int RT_MAX_SECTIONS = 4096;

        // Capacity of the open-scope stack — the DYNAMIC nesting depth,
        // distinct from the distinct-row cap above. Re-entrant call chains
        // (the deposit door re-entering itself through class-update
        // products) push one frame per nested scope even though recursion
        // collapsing re-uses the section rows, so this is sized for deep
        // recursion. Exceeding it trips a Rule-19 assert.
        static constexpr int RT_MAX_OPEN_DEPTH = 1024;
    };

    // Memory-size measurement tunables — only consulted when
    // MEM_MEASUREMENT == 1. See `docs/agentic_swdd/_meta/memory_measurement.md`
    // and the `#define MEM_MEASUREMENT` block at the top of this file.
    struct MemMeasurementParameters {
        // Size of the per-tag byte table. One slot per `LbMemory::ContainerTag`
        // value, including the reserved band holes and the retired tags — the
        // tag IS the index, so the table is sized by the highest tag plus one
        // and never by the count of live containers. Appending a tag past this
        // ceiling trips a Rule-19 assert naming this constant.
        static constexpr int MEM_TAG_SPACE = 1024;

        // Number of worker slots the per-slot accumulator rows are sized for.
        // Each row is MEM_TAG_SPACE int64 counters, so the whole accumulator is
        // MEM_MAX_SLOTS * MEM_TAG_SPACE * 8 bytes of static storage and no heap.
        // Must be at least the run's `logicalCores`; an out-of-range slot
        // asserts rather than wrapping into a neighbour's row.
        static constexpr int MEM_MAX_SLOTS = 64;

        // Structure rows below this share of the peak grand total are folded
        // into a single trailing "(N structures each < M %)" line. 0 shows
        // every row, matching RT_MIN_PERCENTAGE's default.
        static constexpr int MEM_MIN_PERCENTAGE = 0;
    };

}
