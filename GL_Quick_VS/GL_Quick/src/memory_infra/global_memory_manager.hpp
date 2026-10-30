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
#include "../infra/peak_snapshot.hpp"

// Portable force-inline for the paged-memory hot primitives. MSVC's
// per-function inline budget saturates inside the large equi-class apply
// templates, leaving these tiny accessors as real calls on the hottest
// path; the override keeps them folded there. Applies only to leaf
// accessors whose bodies are a few instructions.
#if defined(_MSC_VER)
#define GL_FORCEINLINE __forceinline
#else
#define GL_FORCEINLINE inline __attribute__((always_inline))
#endif

#include <atomic>
#include "extent_file.hpp"

#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace gl {

    /// @brief Extent-file growth chunk: when a new slab passes the file's end,
    ///        the file grows by whole multiples of this (default 256 MiB) so
    ///        growth is coarse, not per-slab.
    constexpr int64_t kExtentGrowChunkBytes = 256LL << 20;

    /// @brief Current and lifetime backing commitment, separate from logical grants.
    /// @details Counters describe physical commitment only and never select proof work.
    struct PoolCommitment {
        int64_t currentBytes = 0;
        int64_t peakBytes = 0;
        int64_t decommittedBytes = 0;
        int64_t recommittedBytes = 0;
        int64_t trimCalls = 0;
    };

    /// @brief Select the independently budgeted memory owner.
    /// @details Main, Persistent, Mail and Lb share the established recycle path.
    /// GpuHost uses a pool-resident first-fit directory for both contiguous ranges
    /// and ordinary arena blocks, with no logical-block eviction machinery.
    /// Each kind names its own exhaustion setting. A private test manager defaults
    /// to Main; enumerators are append-only.
    enum class PoolKind { Main, Persistent, Mail, Lb, GpuHost };

    /// @brief What a granted pool block is used for — the key of a pool's
    ///        purpose attribution.
    ///
    /// @details
    /// The pool itself only knows how many blocks are granted. The memory
    /// playbook needs to know WHICH consumer holds them at the pool's
    /// high-water instant: persistent logic-block state, the per-worker scratch
    /// arenas, the sealed producer-to-consumer pages, or the CUDA route's host
    /// arenas. Every `LbArena` carries one purpose (set before its first block,
    /// `LbArena::setPurpose`) and passes it on every grant and return, so the
    /// manager can keep one in-use counter per purpose under the mutex it
    /// already holds. An arena that never set a purpose counts as
    /// `Unattributed` — a visible residual row, never a silent omission.
    ///
    /// Observability only (Rule 16): no prover, deload or steward decision
    /// reads a purpose counter. Append-only: a new consumer class adds an
    /// enumerator before `Count`.
    ///
    /// @see `GlobalMemoryManager::attribution`, `poolPurposeName`.
    enum class PoolPurpose : uint8_t {
        Unattributed = 0,   ///< Any arena that never set a purpose.
        LbState,            ///< A logic block's own statified state (`LbMemory`, `intToBeProved`).
        ScratchString,      ///< `scratchArenas()` — per-worker string scratch.
        ScratchGen,         ///< `genScratchArenas()` — per-worker container scratch.
        Sealed,             ///< `SealedPageSet` handoff pages (phase-2 records, stumps, deferred drains).
        GpuHost,            ///< CUDA-route host arenas drawn from the pool.
        LbShell,            ///< `LbStore` slots holding the `Memory` node objects.
        Count               ///< Array bound, never a purpose.
    };

    /// @brief Number of `PoolPurpose` values that name a real purpose.
    constexpr int kPoolPurposeCount = static_cast<int>(PoolPurpose::Count);

    /// @brief Returned-block reuse reserve a pool keeps committed when it
    ///        trims (`GlobalMemoryManager::trimRecycledBlocks`).
    ///
    /// @details One value for every trim site — the phase-3 completion trim
    /// of every pool (D-480) and the steward's main-pool trim after a
    /// discharge drain (D-486) — so no site keeps a
    /// different reserve.
    constexpr int64_t kRecycledWarmBytes = int64_t{64} << 20;

    /// @brief Printable name of a pool purpose — the column label of the
    ///        attribution report.
    ///
    /// @details
    /// Stable lowercase identifiers used as `key=value` names in the
    /// diagnostics log and as row labels in the memory log, so a later reader
    /// can grep one purpose across batches.
    ///
    /// @param purpose A real purpose (not `Count`); asserts otherwise.
    /// @return A static string literal.
    const char* poolPurposeName(PoolPurpose purpose);

    /// @brief The prover step a pool grant happened in — the second key of the
    ///        pool attribution.
    ///
    /// @details
    /// The prover sets the current step at its single-threaded seams
    /// (`GlobalMemoryManager::setAttributionPhase`). The manager then keeps, per
    /// step, the highest block count reached while that step was current and
    /// the purpose split at that instant. That answers "in which step does the
    /// pool peak, and what fills it then" without any per-grant timestamp.
    /// `Outside` covers everything not inside a hash-burst iteration (analyzer
    /// construction, counterexample setup, compressor setup, chapter export).
    /// The three CUDA sub-steps split the processor's single phase-2 window into
    /// host projection, device execution with sealing, and the finalize window
    /// that applies the sealed records to the logic blocks.
    ///
    /// Observability only (Rule 16). Append-only before `Count`.
    ///
    /// @see `GlobalMemoryManager::setAttributionPhase`, `poolPhaseName`.
    enum class PoolPhase : uint8_t {
        Outside = 0,        ///< Not inside a hash-burst iteration.
        Phase1,             ///< Mail pull, absorption and standard processing.
        Phase2Execute,      ///< Phase-2 executor window (processor route; CUDA producers).
        Phase2Projection,   ///< CUDA: host projection of a chunk's logic blocks.
        Phase2Device,       ///< CUDA: upload, device execution and host sealing.
        Phase2Finalize,     ///< Phase-2 finalize window (both routes).
        Phase3,             ///< Post-burst standard processing.
        Barrier,            ///< End-of-iteration seams (commit, drains, discharge).
        Count               ///< Array bound, never a phase.
    };

    /// @brief Number of `PoolPhase` values that name a real step.
    constexpr int kPoolPhaseCount = static_cast<int>(PoolPhase::Count);

    /// @brief Printable name of a prover step — the row label of the
    ///        per-step attribution report.
    ///
    /// @param phase A real step (not `Count`); asserts otherwise.
    /// @return A static string literal.
    const char* poolPhaseName(PoolPhase phase);

    /// @brief One consistent snapshot of a pool's purpose and step
    ///        attribution — the value `GlobalMemoryManager::attribution`
    ///        returns.
    ///
    /// @details
    /// All counts are in blocks of the owning pool. `peakBlocks` is the pool's
    /// lifetime high-water; `atPeak` is the purpose split at the instant the
    /// high-water was last raised, so the row sums to `peakBlocks`.
    /// `purposePeak` is each purpose's own high-water, reached at possibly
    /// different instants (never summed). `phasePeak[s]` is the highest block
    /// count reached while step `s` was current, and `phaseAtPeak[s]` its
    /// purpose split (each row sums to its `phasePeak`). `inUse` is the current
    /// split and sums to the current `blocksInUse`.
    ///
    /// @invariant `sum(atPeak) == peakBlocks`; `sum(phaseAtPeak[s]) ==
    ///            phasePeak[s]`; `sum(inUse) == blocksInUse()` at snapshot time.
    struct PoolAttribution {
        int64_t peakBlocks = 0;
        int64_t inUse[kPoolPurposeCount] = {};
        int64_t atPeak[kPoolPurposeCount] = {};
        int64_t purposePeak[kPoolPurposeCount] = {};
        int64_t phasePeak[kPoolPhaseCount] = {};
        int64_t phaseAtPeak[kPoolPhaseCount][kPoolPurposeCount] = {};
    };

    /// @brief Sizing triple of the statification memory hierarchy: one
    ///        program-start pool, carved into equal blocks, carved into equal
    ///        pages.
    ///
    /// @details
    /// The three values mirror the `prover_parameters` config fields
    /// `static_pool_bytes` / `static_block_bytes` / `static_page_bytes` and
    /// must satisfy `isValidStaticMemoryConfig` (pool a whole multiple of
    /// block, block a power of two) and `isValidStaticPageConfig` (block a
    /// whole multiple of page, page a power of two); all strictly positive.
    /// `GlobalMemoryManager::init` asserts that before reserving. `pageBytes`
    /// carries its default so the pool/block-only construction sites compile
    /// unchanged. `kind` selects which config knob the exhaustion assert
    /// names — `static_pool_bytes` (Main), `static_persistent_pool_bytes`
    /// (Persistent), or `static_mail_pool_bytes` (Mail) — and defaults
    /// `PoolKind::Main` so existing construction sites name the main knob
    /// unchanged.
    ///
    /// @invariant Immutable after the manager consumes it — the pool is
    ///            never grown or re-shaped (I-95).
    /// @see `isValidStaticMemoryConfig` / `isValidStaticPageConfig` in
    ///      `parameters.hpp`.
    struct StaticMemoryConfig {
        int64_t poolBytes;
        int32_t blockBytes;
        int32_t pageBytes = 8192;
        PoolKind kind = PoolKind::Main;
    };


    struct SharedSealedBlock;

    /// @brief Exclusive output slice inside a shared physical sealed block.
    /// @details The pool owns allocation metadata; a leased slice belongs to one
    /// filling or sealed output. Its link is private to that output until release.
    /// @invariant Payload never overlaps another slice and stays address-stable.
    struct SealedSlice {
        SharedSealedBlock* block = nullptr;
        SealedSlice* next = nullptr;
        int32_t used = 0;
        int32_t capacity = 0;
        bool leased = false;
    };

    /// @brief Process-wide dispenser of fixed-size memory blocks carved from
    ///        ONE program-start reservation — the root of the statification
    ///        memory hierarchy.
    ///
    /// @details
    /// `init` makes a single reservation of `poolBytes` (the software model
    /// of the ASIC's fixed SRAM): never freed during the run, never grown.
    /// The reservation is address space; each block is committed when it is
    /// first granted. Returned backing beyond the reuse reserve can be
    /// decommitted, so current commitment is distinct from both the carve
    /// high-water and the fixed reservation (I-297).
    /// Per-LB memory managers call `acquireBlock` when they need storage and
    /// `releaseBlock` when an LB deloads to SSD or is destroyed; both run
    /// under an internal mutex because LBs are processed in parallel.
    ///
    /// Grant order under the mutex is nondeterministic. That is sound only
    /// under the statification determinism contract: NO observable behavior
    /// may depend on physical block identity or grant order — containers
    /// address storage through per-LB virtual offsets (resolved to a physical
    /// address only at the access site), and deloaded bytes are a pure
    /// function of logical content (I-107).
    ///
    /// Pool exhaustion is an assert naming the instance's configured pool
    /// knob (`static_pool_bytes` / `static_persistent_pool_bytes` /
    /// `static_mail_pool_bytes` — selected by `StaticMemoryConfig::kind`),
    /// never a fallback heap allocation — sizing problems must surface
    /// immediately, at their origin (I-19, I-95).
    ///
    /// Instantiable as a plain class so unit tests can build private
    /// instances with tiny pools; production code uses THREE process-wide
    /// instances — `staticMemory()` / `initStaticMemory()` (the deloadable
    /// main pool), `persistentMemory()` / `initPersistentMemory()` (the
    /// never-deloaded persistent pool that backs `Memory::intToBeProved`),
    /// and `mailMemory()` / `initMailMemory()` (the never-deloaded mail pool
    /// that backs the cross-LB pull-model mail log).
    ///
    /// @invariant Exactly one reservation per instance lifetime; `init` is
    ///            idempotent for an identical config and asserts on a
    ///            mismatching re-init.
    /// @invariant Every granted block is distinct, lies inside the pool, and
    ///            starts at a whole multiple of `blockBytes` from the pool
    ///            base.
    /// @see `LbArena` (the per-LB consumer), `StaticMemoryConfig`.
    class GlobalMemoryManager {
    public:
        /// @brief Maximum contiguous payload of a shared sealed slice.
        /// @details Sixteen aligned slices share one ordinary pool block. Larger
        /// output allocations retain the existing whole-block byte-arena path.
        /// @return Positive payload capacity for this manager's fixed geometry.
        int32_t sealedSliceCapacity() const;

        /// @brief Lease one exclusive sealed-output slice.
        /// @details A synchronized free list shares physical blocks across sets.
        /// A block remains owned by the pool until its last live slice returns.
        /// @return Empty leased slice whose payload follows its header.
        /// @invariant No heap allocation and no cross-slice payload access.
        SealedSlice* acquireSealedSlice();

        /// @brief Return a slice after its output has been consumed.
        /// @details The caller poisons only its own used payload first. Empty
        /// backing blocks return through ordinary Sealed-purpose accounting.
        /// @param slice Live slice leased from this manager.
        /// @return Nothing.
        /// @invariant Returning one slice never invalidates a live peer.
        void releaseSealedSlice(SealedSlice* slice);

        GlobalMemoryManager() = default;

        /// @brief Returns the reservation (test instances only — the
        ///        process-wide instance lives until process exit).
        ///
        /// @details
        /// Deliberately does NOT assert `blocksInUse() == 0`: at process
        /// exit LBs are reclaimed by the OS rather than walked, so the
        /// process-wide instance legitimately dies with blocks outstanding.
        /// A manager that was never initialized holds no reservation.
        ~GlobalMemoryManager();

        GlobalMemoryManager(const GlobalMemoryManager&) = delete;
        GlobalMemoryManager& operator=(const GlobalMemoryManager&) = delete;
        GlobalMemoryManager(GlobalMemoryManager&&) = delete;
        GlobalMemoryManager& operator=(GlobalMemoryManager&&) = delete;

        /// @brief Make the one reservation and carve it into blocks.
        ///
        /// @details
        /// Asserts `isValidStaticMemoryConfig` on the triple. The pool base
        /// is aligned to 4096 bytes for registered CUDA transfer blocks. The
        /// reservation is address space only; blocks are committed when
        /// first carved (D-418). Idempotent
        /// when called again with a byte-identical config (run modes may
        /// construct more than one `Prover` per process);
        /// asserts on a mismatching re-init — two different sizings in one
        /// process is a configuration bug, not a request to re-shape.
        ///
        /// @param cfg Pool / block / page sizing triple (see
        ///            `StaticMemoryConfig`).
        /// @invariant After return, `initialized()` is true and the pool
        ///            base + shape never change for the instance lifetime.
        void init(const StaticMemoryConfig& cfg);

        /// @brief Whether `init` has run.
        ///
        /// @return `true` after the first successful `init`.
        bool initialized() const;

        /// @brief Lease a contiguous GPU-host range of whole pool blocks.
        /// @details The first fitting free run owns every byte until release.
        /// Directory and alignment overhead are charged to the same reservation.
        /// @param count Positive number of contiguous blocks.
        /// @return Stable block-aligned address; exhaustion asserts.
        /// @invariant Only the independent GPU-host pool permits range grants.
        char* acquireGpuHostRange(int32_t count);

        /// @brief Return an exact GPU-host range after every consumer completes.
        /// @details Head, length and unregistered state must match the live lease.
        /// @param address Original address returned by acquireGpuHostRange.
        /// @param count Original block count.
        /// @return Nothing.
        void releaseGpuHostRange(char* address, int32_t count);

        /// @brief Account registration of an exclusive GPU-host range.
        /// @details Call after successful registration and after unregistering;
        /// a registered range cannot be released or registered twice.
        /// @param address Live range head.
        /// @param count Exact lease length in blocks.
        /// @param pinned True registers accounting; false completes it.
        /// @return Nothing.
        void setGpuHostPinned(char* address, int32_t count, bool pinned);

        /// @brief Measure the largest available contiguous GPU-host run.
        /// @details A sizing query only; it does not allocate or change ownership.
        /// @return Number of contiguous free blocks under the pool mutex.
        int64_t largestGpuHostRun() const;

        /// @brief Return committed bytes in the GPU-host reservation.
        /// @details Includes the ownership directory and recycled committed blocks.
        /// @return Exact committed block count times the block size.
        int64_t gpuHostCommittedBytes() const;

        /// @brief Return currently registered GPU-host bytes.
        /// @details Counts complete exclusive transfer leases.
        /// @return Registered block count times the block size.
        int64_t gpuHostPinnedBytes() const;

        /// @brief Return the peak registered GPU-host storage.
        /// @details Counts whole registered blocks, independently of live leases.
        /// @return Lifetime maximum pinned bytes.
        int64_t gpuHostPeakPinnedBytes() const;

        /// @brief Grant one block (recycled first, else carved fresh).
        ///
        /// @details
        /// Thread-safe (mutex). Recycled blocks are re-granted before fresh
        /// ones so the carve cursor is a high-water mark of simultaneous
        /// demand. Exhaustion — every block carved and none recycled — is an
        /// assert naming `static_pool_bytes`; there is no fallback path.
        ///
        /// The grant is counted under `purpose` for the attribution report
        /// (`attribution`), in the same mutex section as the pool count.
        ///
        /// @param purpose The consumer class the block serves; the returning
        ///                call must pass the same purpose.
        /// @return Pointer to the granted block of `blockBytes()` bytes.
        ///         Physical identity must not influence any observable
        ///         (I-107).
        char* acquireBlock(PoolPurpose purpose = PoolPurpose::Unattributed);

        /// @brief Return a granted block to the recycle queue.
        ///
        /// @details
        /// Thread-safe (mutex). Asserts the pointer lies inside the pool,
        /// is block-aligned, and is currently granted (double release and
        /// foreign pointers are bugs we want at their origin). The return is
        /// subtracted from `purpose`'s in-use count, which must stay
        /// non-negative (a purpose mismatch between grant and return asserts).
        ///
        /// @param block   A pointer previously returned by `acquireBlock`.
        /// @param purpose The purpose the block was granted under.
        void releaseBlock(char* block,
                          PoolPurpose purpose = PoolPurpose::Unattributed);

        /// @brief Grant `count` blocks under ONE mutex acquisition — the bulk
        ///        raw-image reload / eviction grant.
        ///
        /// @details
        /// Thread-safe: takes the pool mutex ONCE, grants `count` blocks
        /// (recycled first, else carved), and updates the accounting
        /// (`blocksInUse` / peak / grant ledger) and the one-shot grant trigger
        /// exactly as `count` separate `acquireBlock` calls would — the trigger
        /// fires once if the batch crosses the armed threshold, outside the
        /// mutex. It exists because the raw arena reload acquires a whole LB's
        /// blocks at once, and the per-block mutex traffic was a measured floor
        /// (~2000 acquire/release round trips per big-LB deload/reload cycle).
        /// Exhaustion of any block asserts naming the instance's pool knob, per
        /// `acquireBlock` (never a fallback, Rule 19). The output array is
        /// filled UNDER the mutex; the caller must NOT re-enter this manager
        /// from within a hypothetical callback — there is none, `out` is a plain
        /// array so directory spill (which does re-enter) happens after the
        /// lock is released.
        ///
        /// @param count   Blocks to grant; >= 1.
        /// @param out     Caller array of at least `count` `char*`; filled
        ///                with the granted block pointers. Physical identity
        ///                must not influence any observable (I-107).
        /// @param purpose The consumer class the blocks serve (attribution).
        void acquireBlocks(int32_t count, char** out,
                           PoolPurpose purpose = PoolPurpose::Unattributed);

        /// @brief Return `count` blocks under ONE mutex acquisition — the bulk
        ///        raw-image eviction / teardown release.
        ///
        /// @details
        /// Thread-safe: takes the pool mutex ONCE and returns every block in
        /// `blocks[0..count)` (each validated as `releaseBlock` would —
        /// in-pool, block-aligned, currently granted — asserts otherwise), then
        /// decrements `blocksInUse` by `count`. The bulk twin of `releaseBlock`
        /// for `LbArena::releaseAll`, so an LB's blocks return with one lock
        /// acquisition instead of one per block — the release half of the
        /// per-block mutex traffic the bulk grant exists to remove.
        ///
        /// @param blocks  Array of `count` pointers previously returned by
        ///                `acquireBlock` / `acquireBlocks`.
        /// @param count   Blocks to return; >= 1.
        /// @param purpose The purpose the blocks were granted under.
        void releaseBlocks(char* const* blocks, int32_t count,
                           PoolPurpose purpose = PoolPurpose::Unattributed);

        /// @brief Set the prover step that subsequent grants are attributed
        ///        to.
        ///
        /// @details
        /// Called by the prover at its single-threaded seams (phase starts,
        /// the CUDA chunk sub-steps, the barrier, and back to `Outside` after
        /// the kernel). Thread-safe (takes the pool mutex), so a worker grant
        /// racing the switch lands in one step or the other, never torn. The
        /// step's high-water starts from the current block count, so a step
        /// that grants nothing still reports the level it ran at.
        ///
        /// Observability only (Rule 16): nothing reads the step back to
        /// decide anything.
        ///
        /// @param phase The step now current; not `Count`.
        void setAttributionPhase(PoolPhase phase);

        /// @brief Snapshot the pool's purpose and step attribution.
        ///
        /// @details
        /// Thread-safe (mutex); returns a copy, so the caller may print it
        /// while grants continue. See `PoolAttribution` for the meaning of
        /// each field and the sum invariants.
        ///
        /// @return The attribution snapshot.
        PoolAttribution attribution() const;

        /// @brief Bytes per block (the grant unit).
        ///
        /// @return `static_block_bytes` of the consumed config. Asserts
        ///         initialized; lock-free (immutable after `init`).
        int32_t blockBytes() const;

        /// @brief Bytes per page (the per-LB allocation unit).
        ///
        /// @details
        /// Owned here so every per-LB manager carves identically; the page
        /// size is a property of the hierarchy, not of one LB.
        ///
        /// @return `static_page_bytes` of the consumed config. Asserts
        ///         initialized; lock-free (immutable after `init`).
        int32_t pageBytes() const;

        /// @brief Total number of blocks the pool holds.
        ///
        /// @return `poolBytes / blockBytes`. Asserts initialized.
        int64_t totalBlocks() const;

        /// @brief Blocks currently granted (telemetry).
        ///
        /// @return Granted minus returned. Thread-safe.
        int64_t blocksInUse() const;

        /// @brief High-water mark of simultaneously granted blocks
        ///        (telemetry for sizing the config defaults).
        ///
        /// @return Peak of `blocksInUse()` over the instance lifetime.
        ///         Thread-safe.
        int64_t peakBlocksInUse() const;

        /// @brief Record a paged container's single -> two-level page-directory
        ///        promotion: bump the promotion count and the peak data-page
        ///        high-water.
        ///
        /// @details
        /// Process documentation only (Rule 16 / I-44): nothing in the prover
        /// reads it. `PagedVector` calls it once when its directory crosses
        /// `dirCap`; `PagedHashIndex` calls it once in `reset` when it builds a
        /// two-level directory. Lock-free (atomics, relaxed) — invoked from the
        /// single-threaded container write side, never the parallel burst. The
        /// count may shift run-to-run with the timing-dependent deload/reload
        /// set, so it is observability, not a determinism-gated value.
        ///
        /// @param numPages Data pages the container holds at the promotion.
        void recordTwoLevelPromotion(int32_t numPages);

        /// @brief Number of single -> two-level page-directory promotions across
        ///        all paged containers this instance has seen.
        ///
        /// @return The promotion count; `> 0` confirms some container's index
        ///         spilled past a single directory page. Lock-free.
        int64_t twoLevelPromotions() const;

        /// @brief Peak data-page count observed at any two-level promotion — the
        ///        deepest paged container this instance saw go two-level.
        ///
        /// @return The high-water data-page count; 0 if none promoted. Lock-free.
        int32_t peakPagesHeld() const;

        /// @brief Print the two-level page-directory telemetry to the common
        ///        diagnostics log — the
        ///        end-of-batch spill summary.
        ///
        /// @details
        /// One line: the promotion count + the peak data-page high-water. A
        /// non-zero count means real containers outgrew one directory page and
        /// spilled into the two-level directory (thousands per batch at the
        /// production 2 KiB page). Observability only.
        void reportPageStats() const;

        /// @brief Grants counted since the last `resetGrantLedger` — the
        ///        steward's only legal mid-iteration pressure signal.
        ///
        /// @details
        /// Within one kernel iteration block traffic is grants-only
        /// (block returns happen at barriers or through the steward's
        /// own planned work), so this counter is monotone and its value
        /// after k grants is independent of grant interleaving — unlike
        /// `blocksInUse()`, which dips as planned frees complete and is
        /// therefore timing-dependent mid-iteration
        /// (I-106). Thread-safe.
        ///
        /// @return Grants since the last ledger reset.
        int64_t grantsSinceBarrier() const;

        /// @brief Zero the grant ledger — called at the kernel's
        ///        end-of-iteration barrier, single-threaded.
        void resetGrantLedger();

        /// @brief Arm the once-per-iteration grant trigger: `fire` runs
        ///        when the ledger reaches `threshold`, then self-disarms.
        ///
        /// @details
        /// The crossing grant invokes `fire` AFTER releasing the
        /// manager's mutex, exactly once (the trigger disarms before the
        /// call). Determinism: the iteration's grant SET is
        /// deterministic, so whether the ledger reaches `threshold` in a
        /// given iteration is run-invariant — only WHICH thread's grant
        /// crosses varies, which is unobservable. `fire` must be
        /// lock-light (the steward's: set an atomic flag + notify) and
        /// must not call back into this manager. Asserts not already
        /// armed (re-arming without a barrier reset is a lifecycle bug)
        /// and a non-empty callable.
        ///
        /// @param threshold Ledger value that fires the trigger; >= 1
        ///                  (a zero-or-negative demand is decided at the
        ///                  barrier itself, never delegated here).
        /// @param fire      Callback run once by the crossing grant.
        void armGrantTrigger(int64_t threshold,
                             std::function<void()> fire);

        /// @brief Disarm the grant trigger — barrier cleanup; defined
        ///        no-op when the trigger already fired or was never
        ///        armed this iteration (both are completed lifecycles,
        ///        not failures).
        void disarmGrantTrigger();

        /// @brief Install the grant-time pressure valve: the pager's
        ///        backpressure on block grants.
        ///
        /// @details
        /// Before a grant would leave fewer than `floorBlocks` blocks free,
        /// the granting thread calls `valve` once, outside the pool mutex,
        /// and grants afterwards whatever the valve achieved: when nothing
        /// could be freed, a genuinely full pool still asserts at the wall
        /// (Rule 19). The production valve evicts one idle logic block behind
        /// the open window's cursor (`MemorySteward::relieveGrantPressure`),
        /// so growth inside a claimed logic block is throttled to the
        /// eviction rate instead of outrunning it
        /// (D-422). A valve that grants from this pool
        /// itself does not re-enter the valve (per-thread guard). An empty
        /// `valve` clears the hook. Installation and clearing happen on one
        /// thread while no other thread grants from this pool.
        ///
        /// @param valve Pressure relief to run before a grant below the floor;
        ///              empty clears.
        /// @param floorBlocks Free blocks a grant must leave; ignored when
        ///                    `valve` is empty.
        /// @return Nothing.
        void setPressureValve(std::function<void()> valve, int64_t floorBlocks);

        /// @brief Install (or clear, with an empty function) the FORENSIC
        ///        EXHAUSTION REPORTER — a diagnostic callback `grantLocked`
        ///        invokes at the wall, immediately BEFORE the exhaustion
        ///        assert fires.
        ///
        /// @details
        /// Pure observability at the point of death (the stuck-tripwire
        /// precedent: print, then assert — the assert REMAINS the failure,
        /// Rule 19 untouched; no fallback path is created). The prover
        /// installs it for the `prove()` scope (capturing the LB grid and
        /// the steward) and clears it on scope exit; a pool with no
        /// reporter installed asserts exactly as before. A `std::function`
        /// is deliberate: installation is a cold, single-threaded,
        /// once-per-prove operation, never on a grant path.
        ///
        /// CONTRACT — THE REPORTER RUNS WHILE THE POOL MUTEX IS HELD. It
        /// must never call back into this manager: no `acquireBlock` /
        /// `releaseBlock` / `acquireBlocks` / `releaseBlocks`, and none of
        /// the mutex-taking getters (`blocksInUse()`,
        /// `grantsSinceBarrier()`, `assignDeloadOrdinal()`, ...) — any of
        /// those deadlocks on the held mutex. The pool counters it needs
        /// are therefore PASSED IN (`blocksInUse`, `totalBlocks`, read
        /// under the already-held mutex at the call site). Legal reads
        /// inside the reporter: per-LB claim words (atomics), per-arena
        /// bookkeeping (`LbArena::blocksHeld()` / `resident()` — plain
        /// member reads that never touch this manager), the steward's
        /// relaxed in-flight atomics, `DeloadStats` atomics, and the
        /// scratch registries' per-slot arena bookkeeping. Reads of other
        /// threads' live arenas are RACY (the process is about to abort;
        /// the census is approximate by design).
        ///
        /// @param reporter Callback receiving (blocksInUse, totalBlocks) at
        ///                 the wall; empty clears the hook (a completed
        ///                 lifecycle, not a failure).
        void setExhaustionReporter(
            std::function<void(int64_t, int64_t)> reporter);

        /// @brief Invoke the installed exhaustion reporter under the pool
        ///        mutex — the unit-test seam for the death-path call
        ///        context.
        ///
        /// @details
        /// The real invocation site is `grantLocked` at the wall, followed
        /// by an assert that aborts the process — untestable in the
        /// in-tree harness (no death-test support; `assert` aborts the
        /// whole suite). This seam reproduces the exact calling context
        /// (pool mutex held, counters read under it) WITHOUT the abort, so
        /// a test can prove the census callback performs no manager
        /// re-entry (a violation deadlocks the test loudly) and receives
        /// the correct counter values. Asserts a reporter is installed —
        /// invoking a missing diagnostic is a test bug (production guards
        /// the call on installation).
        void invokeExhaustionReporterForTest();

        /// @brief Assign the next process-monotonic deload ordinal — the
        ///        injective per-LB file-name handle that replaced the
        ///        chain hash.
        ///
        /// @details
        /// Thread-safe (mutex). Post-increments one counter, so two LBs
        /// can never receive the same ordinal in one process — file
        /// identity is provably unique, no hash collision possible. The
        /// counter is process-monotonic and deliberately survives
        /// `resetDeloadRegistry` (a persisting LB keeps its ordinal across
        /// batches; reusing a number could collide a stale image with a
        /// freshly assigned one). A fresh process starts at 0, so two runs
        /// of the same input assign the same ordinals — the kernel barrier
        /// hands LBs to assignment in a deterministic order.
        ///
        /// @return The freshly assigned ordinal; >= 0, unique this process.
        int64_t assignDeloadOrdinal();

        /// @brief Record an `ordinal -> full LB chain` mapping for
        ///        `registry.txt` — the human/audit resolver of the numeric
        ///        file names.
        ///
        /// @details
        /// Thread-safe (mutex) — the parallel synchronous-eviction sweep
        /// registers from several worker threads. Idempotent: re-recording
        /// the same ordinal with the same chain is the normal case (every
        /// dump re-registers); recording an ordinal already bound to a
        /// DIFFERENT chain is an identity bug and asserts (Rule 19). The
        /// ordinal must already be assigned (`< nextDeloadOrdinal_`).
        ///
        /// @param ordinal A value returned by `assignDeloadOrdinal`.
        /// @param chain   The LB's full parent-chain string (its identity).
        void registerDeloadOrdinal(int64_t ordinal,
                                   const std::string& chain);

        /// @brief The accumulated `ordinal -> chain` map — the input to
        ///        `lbdeload::rewriteRegistry`.
        ///
        /// @details
        /// Returns a reference, NOT a copy: the sole caller is the kernel
        /// barrier fold, single-threaded after the steward quiesce and the
        /// synchronous-eviction join, so no concurrent mutation can race
        /// the read. `std::map` keeps the dump ascending by ordinal
        /// (deterministic output).
        ///
        /// @return The registry map (ascending ordinal).
        const std::map<int64_t, std::string>& deloadRegistry() const;

        /// @brief Clear the registry map at batch start so `registry.txt`
        ///        describes only the current `.deload/` contents.
        ///
        /// @details
        /// Clears the `ordinal -> chain` map (called next to
        /// `purgeDeloadDirectory`, which empties `.deload/`). Deliberately
        /// does NOT reset the ordinal counter: the counter stays
        /// process-monotonic so a persisting LB never collides a reused
        /// number (see `assignDeloadOrdinal`). Every LB that dumps in the
        /// new batch re-registers, so the cleared map refills completely.
        void resetDeloadRegistry();

        /// @brief Open (creating) the ONE extent file for the v4 raw-eviction
        ///        images and arm the extent datapath for this batch.
        ///
        /// @details
        /// Closes any prior batch's extent file, opens the new one,
        /// preallocates it to `initialBytes`, (re)initializes the slab
        /// allocator to the pool block granularity, and BUMPS the extent
        /// EPOCH so any LB carrying a slab from a purged prior batch is treated
        /// as slab-less (its stale `rawExtentOffset_` is ignored — offsets
        /// recycled from 0 at reset). Sets `useExtent()` true. Called at batch
        /// start (after `purgeDeloadDirectory` deletes the old file). Only when
        /// `parameters.enable_extent_deload`; otherwise the named-file raw path
        /// stays active.
        ///
        /// @param path         The extent file path (under the deload dir).
        /// @param initialBytes Preallocated size (default 1.5x the pool).
        void openExtentFile(const std::filesystem::path& path,
                            int64_t initialBytes);

        /// @brief Close the extent file (if open) and disarm the datapath.
        ///
        /// @details
        /// Called BEFORE `purgeDeloadDirectory` (which deletes the file — an
        /// open handle would block the delete on Windows) and at process
        /// teardown via the destructor. Sets `useExtent()` false. A no-op when
        /// already closed.
        void closeExtentFile();

        /// @brief Whether the extent datapath is armed this batch.
        /// @return True between `openExtentFile` and `closeExtentFile`.
        bool useExtent() const {
            return extentEnabled_.load(std::memory_order_relaxed);
        }

        /// @brief The current extent epoch — bumped at every `openExtentFile`.
        ///
        /// @details
        /// An LB's cached slab (`rawExtentOffset_`) is valid ONLY if the LB's
        /// `rawExtentEpoch_` equals this; a mismatch means the slab is from a
        /// purged batch (offsets since recycled from 0) and must be ignored.
        /// Read on the dump path; only mutated single-threaded at batch start.
        ///
        /// @return The epoch id.
        int64_t extentEpoch() const { return extentEpoch_; }

        /// @brief The open extent file (positioned I/O at disjoint offsets).
        /// @return A reference for the raw dump/load positioned calls.
        PositionedFile& extentFile() { return extentFile_; }

        /// @brief Allocate an extent slab for an image of `need` bytes, growing
        ///        the file if the high-water passed its end.
        ///
        /// @details
        /// Serializes the slab allocation AND the file grow under one mutex so
        /// the `PositionedFile::grow` targets stay monotone under concurrent
        /// evictions (two threads racing the high-water must not order two
        /// grows backwards). The positioned WRITE that follows runs OUTSIDE
        /// this lock (disjoint offsets). Records the slab under its ordinal in
        /// the slab registry (the `registry.txt` extent audit columns).
        ///
        /// @param ordinal The LB's deload ordinal (slab-registry key).
        /// @param need    The image byte count to hold.
        /// @return The slab offset and its class capacity.
        SlabAllocation allocateExtentSlab(int64_t ordinal, int64_t need);

        /// @brief Return an extent slab to its class free-list and drop its
        ///        slab-registry row (class promotion / LB discharge).
        ///
        /// @param ordinal    The owning LB's deload ordinal (registry key).
        /// @param offset     The slab offset from `allocateExtentSlab`.
        /// @param classBytes The slab's class capacity.
        void freeExtentSlab(int64_t ordinal, int64_t offset,
                            int64_t classBytes);

        /// @brief The per-ordinal slab placements — the extent audit columns of
        ///        `lbdeload::rewriteRegistry`.
        ///
        /// @details
        /// Returns a reference, NOT a copy, on the same justification as
        /// `deloadRegistry()`: the sole production caller is the kernel barrier
        /// fold, single-threaded after the steward quiesce, so no executor
        /// mutation races the read. Reload never consults it (the LB's own
        /// `rawExtentOffset_` is the source of truth); it exists for the
        /// human-audit registry file only.
        ///
        /// @return Ordinal → current slab (ascending ordinal).
        const std::map<int64_t, SlabAllocation>& extentSlabRegistry() const {
            assert(extentSlabRegistry_.has_value());
            return *extentSlabRegistry_;
        }

        /// @brief Σ live slab class sizes (the extent internal-slack numerator).
        /// @return `extentAllocatedBytes` telemetry.
        int64_t extentAllocatedBytes() const {
            return extentAllocator_.allocatedBytes();
        }

        /// @brief The extent file's current logical size.
        /// @return `extentFileBytes` telemetry.
        int64_t extentFileBytes() const { return extentFile_.fileSize(); }

        /// @brief Adjust the extent live-occupancy total by a signed delta.
        ///
        /// @details
        /// The dump wiring calls this with `newImage - lastImage` on each
        /// extent dump (the LB's occupancy replaces its prior) and `-lastImage`
        /// on slab free. Batch-scoped: zeroed by `openExtentFile`, so it stays
        /// correct across the several prove() calls of one batch (unlike the
        /// per-prove `DeloadStats` reset). One relaxed accumulation.
        ///
        /// @param delta Signed byte delta.
        void addExtentLive(int64_t delta) {
            extentLiveBytes_.fetch_add(delta, std::memory_order_relaxed);
        }

        /// @brief Σ image bytes of currently-Dumped raw LBs (the slack numerator).
        /// @return `extentLiveBytes` telemetry.
        int64_t extentLiveBytes() const {
            return extentLiveBytes_.load(std::memory_order_relaxed);
        }


        /// @brief Release excess returned blocks while preserving recycle order.
        ///
        /// @details
        /// The first warmBytes/blockBytes entries of the existing recycle queue
        /// are retained for reuse. Later committed entries are decommitted under
        /// the pool mutex, coalescing adjacent addresses in queue order. Granted
        /// blocks, virtual addresses, carve limits and grant accounting stay fixed.
        /// Already decommitted entries remain reserved until their next grant.
        /// This operation is restricted to the four original pools; GPU-host
        /// backing stays committed for reuse.
        ///
        /// @param warmBytes Nonnegative whole-block reuse reserve in bytes.
        /// @return Number of backing bytes decommitted by this call.
        /// @invariant Only exclusively returned whole-page blocks are decommitted.
        /// @see commitment, grantLocked
        int64_t trimRecycledBlocks(int64_t warmBytes);

        /// @brief Return one synchronized commitment-counter snapshot.
        ///
        /// @details Current commitment includes both granted and retained free
        /// blocks. The lifetime peak and cumulative release/recommit totals have
        /// separate denominators and are never read to select inference work.
        /// GPU-host commitment only grows, so current equals peak and its
        /// decommit, recommit and trim totals remain zero.
        ///
        /// @return Current and lifetime commitment counters; all zero before init.
        /// @see trimRecycledBlocks
        PoolCommitment commitment() const;

        /// @brief Observe committed pool blocks and their exact grant state.
        /// @details Diagnostic only; callers have stopped all allocation writers.
        /// A recycled block is retained memory, not live container content.
        /// @return Nothing.
        void observeResidentRanges() {
            std::lock_guard<std::mutex> lock(mutex_);
            if (pool_ == nullptr) return;
            if (cfg_.kind == PoolKind::GpuHost) {
                for (int64_t i = 0; i < totalBlocks_; ++i) {
                    if (!gpuHostDirectory_[i].committed) continue;
                    peak_snapshot::Scope owner(gpuHostDirectory_[i].run != 0
                        ? "unassignedGrantedBlocks" : "recycledBlocks");
                    peak_snapshot::note(pool_ + i * cfg_.blockBytes,
                        static_cast<std::size_t>(cfg_.blockBytes), 4);
                }
                return;
            }
            for (int64_t i=0;i<carveCursor_;++i) {
                if (!committed_[static_cast<std::size_t>(i)]) continue;
                peak_snapshot::Scope owner(granted_[static_cast<std::size_t>(i)]
                    ? "unassignedGrantedBlocks" : "recycledBlocks");
                peak_snapshot::note(pool_+i*cfg_.blockBytes,
                    static_cast<std::size_t>(cfg_.blockBytes),4);
            }
        }

    private:
        /// @brief Dequeue-or-carve one block and flag it granted; the
        ///        accounting-free core shared by both grant paths. Caller
        ///        holds `mutex_`.
        ///
        /// @details
        /// Fresh and decommitted recycled blocks commit before use; already
        /// committed recycled blocks retain their backing.
        ///
        /// @param purpose Owning accounting category.
        /// @return The granted block.
        char* grantLocked(PoolPurpose purpose);

        /// @brief Run the pressure valve when a grant of `count` blocks would
        ///        leave fewer free blocks than the valve's floor.
        ///
        /// @details
        /// Lock-free check on `blocksInUseHint_`; below the floor, copies the
        /// valve under `mutex_` and calls it outside the mutex, unless this
        /// thread is already inside a valve call (the re-entry is the outer
        /// call's own relief). No valve installed is a defined no-op.
        ///
        /// @param count Blocks the caller is about to grant.
        /// @return Nothing.
        void relievePressureBeforeGrant(int64_t count);

        /// @brief Validate and un-flag a returned block, park it on the
        ///        recycle queue; the accounting-free core shared by both
        ///        release paths. Caller holds `mutex_`.
        ///
        /// @param block The block being returned.
        /// @param purpose Original accounting category.
        /// @return Nothing.
        void returnLocked(char* block, PoolPurpose purpose);

        /// @brief Book `count` fresh grants under `purpose`: raise the pool,
        ///        purpose and step high-waters and snapshot the purpose split
        ///        at each new high. Caller holds `mutex_` and has already
        ///        advanced `blocksInUse_`.
        ///
        /// @param purpose The purpose of the granted blocks.
        /// @param count   Blocks just granted; >= 1.
        void noteGrantLocked(PoolPurpose purpose, int64_t count);

        /// @brief Allocate a first-fit run while holding the pool mutex.
        /// @details Used by both single-block and contiguous GPU-host grants.
        /// @param count Positive run length.
        /// @param purpose Owning accounting category.
        /// @return Committed stable address; exhaustion asserts.
        char* grantGpuHostLocked(int32_t count, PoolPurpose purpose);

        /// @brief Validate and free one exact GPU-host lease under the mutex.
        /// @details Neither live transfer registration nor a partial lease may end.
        /// @param address Original lease head.
        /// @param count Exact original length.
        /// @param purpose Original accounting category.
        /// @return Nothing.
        void returnGpuHostLocked(char* address, int32_t count, PoolPurpose purpose);

        struct GpuHostBlock {
            int32_t run = 0;
            uint8_t committed = 0;
            uint8_t purpose = 0;
            uint8_t pinned = 0;
            uint8_t padding = 0;
        };
        GpuHostBlock* gpuHostDirectory_ = nullptr;
        int64_t gpuHostDirectoryBlocks_ = 0;
        int64_t gpuHostCommittedBlocks_ = 0;
        int64_t gpuHostPinnedBlocks_ = 0;
        int64_t gpuHostPeakPinnedBlocks_ = 0;

        mutable std::mutex mutex_;
        std::mutex sealedMutex_;
        SharedSealedBlock* sealedAvailable_ = nullptr;
        char* pool_ = nullptr;
        StaticMemoryConfig cfg_{ 0, 0, 0 };
        int64_t totalBlocks_ = 0;
        int64_t carveCursor_ = 0;          // next never-granted block index
        std::optional<std::deque<char*>> recycled_;       // returned blocks awaiting re-grant
        std::vector<bool> granted_;        // per-block grant flag (assert aid)
        std::vector<bool> committed_;      // backing commitment, independent of ownership
        PoolCommitment commitment_{};
        int64_t blocksInUse_ = 0;
        // Relaxed copy of blocksInUse_ for the lock-free valve check, written
        // under mutex_ wherever blocksInUse_ changes.
        std::atomic<int64_t> blocksInUseHint_{ 0 };
        // Grant-time pressure valve (D-422): empty =
        // none installed; read under mutex_, called outside it.
        std::function<void()> pressureValve_;
        std::atomic<int64_t> pressureFloorBlocks_{ -1 };
        int64_t peakBlocksInUse_ = 0;
        // Purpose / step attribution (process documentation, Rule 16; never a
        // prover input). Guarded by mutex_, updated in the grant and return
        // sections that already hold it.
        PoolPhase currentPhase_ = PoolPhase::Outside;
        PoolAttribution attribution_{};
        // Two-level page-directory telemetry (process documentation, Rule 16 /
        // I-44; never a prover input). Atomic so the single-threaded container
        // write side records without taking the pool mutex.
        std::atomic<int64_t> twoLevelPromotions_{ 0 };
        std::atomic<int32_t> peakPagesHeld_{ 0 };
        int64_t grantsSinceBarrier_ = 0;   // monotone within an iteration
        int64_t grantTriggerThreshold_ = -1;  // -1 = disarmed
        std::function<void()> grantTriggerFire_;
        // Forensic exhaustion reporter (diagnostic only): invoked by
        // grantLocked at the wall, under mutex_, BEFORE the exhaustion
        // assert; empty = not installed. See setExhaustionReporter.
        std::function<void(int64_t, int64_t)> exhaustionReporter_;
        int64_t nextDeloadOrdinal_ = 0;    // process-monotonic deload file id
        std::optional<std::map<int64_t, std::string>> deloadRegistry_;  // ordinal -> chain

        // v4 raw-eviction extent file (D-195 extent
        // datapath). ONE preallocated file per batch; each LB owns a slab.
        // extentMutex_ serializes slab alloc + file grow (grow must stay
        // monotone); the positioned writes/reads run outside it (disjoint
        // offsets). extentEnabled_ is the A/B gate (relaxed-read on the dump
        // path); extentEpoch_ invalidates cross-batch stale slabs.
        PositionedFile extentFile_;
        ExtentAllocator extentAllocator_;
        std::mutex extentMutex_;
        std::atomic<bool> extentEnabled_{ false };
        int64_t extentEpoch_ = 0;
        std::atomic<int64_t> extentLiveBytes_{ 0 };
        // ordinal -> current slab, for registry.txt's audit columns only
        // (reload uses the LB's own rawExtentOffset_). Under extentMutex_.
        std::optional<std::map<int64_t, SlabAllocation>> extentSlabRegistry_;
    };

    /// @brief The process-wide static-memory manager instance.
    ///
    /// @details
    /// Production binding for `Memory` / per-LB managers. Unit tests build
    /// private `GlobalMemoryManager` instances instead and leave this one
    /// to the harness-level init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initStaticMemory` before any block traffic).
    GlobalMemoryManager& staticMemory();

    /// @brief Initialize the process-wide manager from a config triple.
    ///
    /// @details
    /// Thin forwarder to `staticMemory().init(cfg)`; carries the same
    /// idempotent-same-config / assert-on-mismatch contract. Called by the
    /// `Prover` constructor right after the JSON parameter
    /// parse, and by the unit-test harness with a small test pool.
    ///
    /// @param cfg Pool / block / page sizing triple.
    void initStaticMemory(const StaticMemoryConfig& cfg);

    /// @brief The process-wide PERSISTENT static-memory manager instance — the
    ///        second pool that backs `Memory::intToBeProved`.
    ///
    /// @details
    /// A separate reservation from `staticMemory()`, with smaller blocks
    /// (`static_persistent_block_bytes`). Each LB's persistent `LbArena` draws
    /// from this pool and is NEVER deloaded — its content stays resident for the
    /// LB's whole active life and is reclaimed only at discharge — so the
    /// deactivation survey can read `intToBeProved` regardless of the main
    /// arena's deload state (the determinism fix). Distinct pool, distinct
    /// telemetry, distinct exhaustion assert (`static_persistent_pool_bytes`).
    /// Unit tests build private instances and leave this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initPersistentMemory` before any persistent block traffic).
    /// @see `staticMemory` (the deloadable main pool), `initPersistentMemory`.
    GlobalMemoryManager& persistentMemory();

    /// @brief Initialize the process-wide persistent manager from a config
    ///        triple (must carry `isPersistent == true`).
    ///
    /// @details
    /// Thin forwarder to `persistentMemory().init(cfg)`; same idempotent-same-
    /// config / assert-on-mismatch contract as `initStaticMemory`. Called by the
    /// `Prover` constructor right after `initStaticMemory`, and by
    /// the unit-test harness with a small test pool.
    ///
    /// @param cfg Persistent pool / block / page sizing triple; its `kind`
    ///            must be `PoolKind::Persistent` so the exhaustion assert
    ///            names the right knob.
    void initPersistentMemory(const StaticMemoryConfig& cfg);

    /// @brief The process-wide MAIL static-memory manager instance — the
    ///        third pool that backs the cross-LB pull-model mail log.
    ///
    /// @details
    /// A separate reservation from both `staticMemory()` and
    /// `persistentMemory()`, NEVER deloaded: it holds every LB's committed
    /// mail batches and the per-(recipient, ancestor) ingestion cursors for a
    /// whole execution batch. A stand-alone pool by design — nothing reads its
    /// grant ledger, so it plays no role in any deload / throttle / steward
    /// decision, and the mail content never competes with the deloadable main
    /// pool. The single `Prover`-owned `mailArena` draws blocks
    /// from it; the `MailLog` containers pack pages out of that arena. Distinct
    /// pool, distinct telemetry, distinct exhaustion assert
    /// (`static_mail_pool_bytes`). Unit tests build private instances and leave
    /// this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initMailMemory` before any mail block traffic).
    /// @see `staticMemory` (the deloadable main pool), `persistentMemory`
    ///      (the second pool), `initMailMemory`.
    GlobalMemoryManager& mailMemory();

    /// @brief Initialize the process-wide mail manager from a config triple
    ///        (must carry `kind == PoolKind::Mail`).
    ///
    /// @details
    /// Thin forwarder to `mailMemory().init(cfg)`; same idempotent-same-config
    /// / assert-on-mismatch contract as `initStaticMemory`. Asserts the config
    /// carries `PoolKind::Mail` so the exhaustion assert names the right knob.
    /// Called by the `Prover` constructor right after
    /// `initPersistentMemory`, and by the unit-test harness with a small test
    /// pool.
    ///
    /// @param cfg Mail pool / block / page sizing triple; `kind` must be
    ///            `PoolKind::Mail`.
    void initMailMemory(const StaticMemoryConfig& cfg);

    /// @brief The process-wide LB-BODY static-memory manager instance — the
    ///        fourth pool that backs the LB object store.
    ///
    /// @details
    /// A separate reservation from the three pools above, NEVER deloaded: it
    /// backs the `LbStore` on the `Prover`, which places every
    /// `Memory` node object into a fixed-size slot carved from this pool's
    /// blocks instead of the malloc heap. A stand-alone pool by design —
    /// nothing reads its grant ledger, so it plays no role in any deload /
    /// throttle / steward decision, and the shells never compete with the
    /// deloadable main pool. The `LbStore` carves each block into
    /// `sizeof(Memory)`-sized slots (it draws blocks, not pages). Distinct pool,
    /// distinct telemetry, distinct exhaustion assert (`static_lb_pool_bytes`).
    /// Unit tests build private instances and leave this one to the harness init.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initLbMemory` before any LB-store block traffic).
    /// @see `staticMemory` (the deloadable main pool), `persistentMemory`
    ///      (the second pool), `mailMemory` (the third pool), `initLbMemory`.
    GlobalMemoryManager& lbMemory();

    /// @brief Initialize the process-wide LB-body manager from a config triple
    ///        (must carry `kind == PoolKind::Lb`).
    ///
    /// @details
    /// Thin forwarder to `lbMemory().init(cfg)`; same idempotent-same-config
    /// / assert-on-mismatch contract as `initStaticMemory`. Asserts the config
    /// carries `PoolKind::Lb` so the exhaustion assert names the right knob.
    /// Called by the `Prover` constructor right after
    /// `initMailMemory`, and by the unit-test harness with a small test pool.
    ///
    /// @param cfg LB-body pool / block / page sizing triple; `kind` must be
    ///            `PoolKind::Lb`.
    void initLbMemory(const StaticMemoryConfig& cfg);

    /// @brief Access the separately initialized GPU-host pool.
    /// @details Access alone reserves nothing; initialization precedes allocation.
    /// @return Process-wide fifth static-memory manager.
    GlobalMemoryManager& gpuHostMemory();

    /// @brief Initialize an independent GPU-host reservation.
    /// @details Uses the normal immutable-shape initialization contract.
    /// @param cfg Positive valid geometry with kind GpuHost.
    /// @return Nothing.
    void initGpuHostMemory(const StaticMemoryConfig& cfg);

}
