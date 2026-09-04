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

#include "global_memory_manager.hpp"
#include "lb_arena.hpp"

#include <cassert>
#include <cstdint>
#include <deque>

namespace gl {

    /// @brief The per-worker scratch string arena IS the shared bump arena
    ///        (`LbArena`) on the COLD grant path — `ScratchArena` is an alias,
    ///        not a second implementation.
    ///
    /// @details
    /// One management code, one substrate. A `ScratchArena` is an `LbArena`
    /// default-constructed unbound and then `bind`-ed to the pool (the cold
    /// `acquireBlock` grant path, the cold poison byte, no reserve cap). The
    /// scratch consumers use the page-tier surface: `allocBytes` (a resolved
    /// `char*`, the fill-then-wrap pattern), `mark` / `rewind` (the per-call
    /// window), and `releaseAll` (the per-task wholesale reclaim — RETURNS the
    /// blocks to the pool and bumps the generation so stale views assert). A
    /// scratch arena is never deload-registered and never compacted while a
    /// task holds it — it is filled and then fully released within one worker
    /// task — so a resolved pointer stays valid until its rewind / releaseAll,
    /// the guarantee the scratch string views rely on.
    ///
    /// @invariant Scratch blocks are RELEASED the moment a worker task ends
    ///            (no retention); allocation never straddles a page; freed
    ///            spans are poisoned with `kArenaPoisonByte`.
    /// @invariant Arena content is never an input to any observable behavior;
    ///            only the bytes of individual live allocations are read, and
    ///            only within their generation.
    /// @see `LbArena` (the shared implementation), `ScratchArenaRegistry` (the
    ///      per-slot owner), `GlobalMemoryManager::acquireBlock`.
    using ScratchArena = LbArena;

    /// @brief Slot-indexed owner of the per-worker scratch arenas.
    ///
    /// @details
    /// "Per thread" in the string statification model is realised as per
    /// WORKER SLOT (the `coreId` every executor entry point already
    /// receives), not as `thread_local`: the prover's pools spawn fresh
    /// `std::thread`s every phase, so thread-local arenas would construct and
    /// destruct each phase. Per-slot keeps the arena OBJECTS stable (bound
    /// once at init); only their blocks cycle — acquired lazily on first use,
    /// released back to the pool at each worker task's end. One slot runs at
    /// most one executor at a time, phases join before the next starts, so
    /// slot ownership gives the same isolation a private thread arena would
    /// (D-163).
    ///
    /// `init` is single-threaded (analyzer construction); `forSlot` after
    /// init is a read of stable storage (`std::deque` — addresses never
    /// move) and safe from any worker.
    ///
    /// @invariant Slot count is fixed at first init; a mismatching re-init
    ///            asserts (one process, one shape).
    /// @see `ScratchArena`, `scratchArenas()` / `initScratchArenas()` (the
    ///      process-wide binding).

    class ScratchArenaRegistry {
    public:
        /// @brief Create `slotCount` cold scratch arenas bound to `global`.
        ///
        /// @details
        /// Each slot is default-constructed unbound and then `bind`-ed cold
        /// (no reserve cap — the pool's own exhaustion assert is the backstop,
        /// and the per-task release keeps the footprint transient). Idempotent
        /// for an identical shape (run modes construct several
        /// `ExpressionAnalyzer`s per process); asserts on a mismatching
        /// re-init.
        ///
        /// @param global    Pool the arenas draw blocks from.
        /// @param slotCount Number of worker slots (`logicalCores`).
        void init(GlobalMemoryManager* global, unsigned slotCount);

        /// @brief Whether `init` has run.
        ///
        /// @return `true` after the first successful `init`.
        bool initialized() const { return !arenas_.empty(); }

        /// @brief Number of slots fixed at init.
        ///
        /// @return Slot count. Asserts initialized.
        unsigned slotCount() const {
            assert(initialized());
            return static_cast<unsigned>(arenas_.size());
        }

        /// @brief The arena owned by one worker slot.
        ///
        /// @param slot Worker slot id (`coreId`), `< slotCount()`.
        /// @return That slot's arena; stable address for the registry
        ///         lifetime.
        ScratchArena& forSlot(unsigned slot) {
            assert(initialized());
            assert(slot < arenas_.size()
                && "scratch arena slot out of range — coreId beyond the "
                   "logicalCores the registry was initialized with");
            return arenas_[slot];
        }


    private:
        std::deque<ScratchArena> arenas_;  // deque: stable element addresses
        GlobalMemoryManager* global_ = nullptr;
    };

    /// @brief The process-wide scratch-arena registry.
    ///
    /// @details
    /// Production binding for the prover's worker slots. Unit tests build
    /// private `ScratchArenaRegistry` / `ScratchArena` instances over private
    /// pools instead.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initScratchArenas` before any scratch allocation).
    ScratchArenaRegistry& scratchArenas();

    /// @brief Initialize the process-wide registry against the process-wide
    ///        pool.
    ///
    /// @details
    /// Thin forwarder to `scratchArenas().init(&staticMemory(), ...)`; carries
    /// the same idempotent-same-shape / assert-on-mismatch contract. Called
    /// by the `ExpressionAnalyzer` constructor after `initStaticMemory` and
    /// after `logicalCores` is known.
    ///
    /// @param slotCount Number of worker slots (`logicalCores`).
    void initScratchArenas(unsigned slotCount);

    /// @brief The process-wide request-generation scratch-arena registry — a
    ///        SECOND per-slot set of arenas, distinct from `scratchArenas()`.
    ///
    /// @details
    /// The phase-2 request generators need scratch that the string-scratch
    /// arena (`scratchArenas()`) cannot host: the depth-first frontier wants
    /// byte-bump `popTo` reclaim, and the per-batch containers
    /// (`baseCandidates` / `sortedPairs` / the dedup) want the page tier — but a
    /// `ScratchScope` rewind on the string arena would free page-tier pages, and
    /// a byte-bump there would muddy the `ScratchString` liveness tripwire (both
    /// key off the shared `usedBytes()`). A separate per-slot arena keeps the two
    /// concerns isolated: this one mixes byte-bump (the DFS stack via
    /// `ArenaStack`) and page tier (the append containers) exactly the way a
    /// per-LB arena already does, while the string arena stays pure. Released per
    /// worker task alongside the string arena at `performElem2` exit.
    ///
    /// @return The singleton (constructed on first use; `init` it via
    ///         `initGenScratchArenas` before any request-generation scratch).
    /// @see `scratchArenas()` (the string-scratch sibling), `ArenaStack`.
    ScratchArenaRegistry& genScratchArenas();

    /// @brief Initialize the process-wide request-generation registry against
    ///        the process-wide pool.
    ///
    /// @details
    /// Thin forwarder to `genScratchArenas().init(&staticMemory(), ...)`; same
    /// idempotent-same-shape / assert-on-mismatch contract as
    /// `initScratchArenas`. Called by the `ExpressionAnalyzer` constructor right
    /// after `initScratchArenas`, with the same `logicalCores + 1` slot count:
    /// the last slot is the reserved single-threaded slot every
    /// `g_currentCoreId == -1` caller resolves to (`currentGenSlot()`), so no
    /// worker's arena or rule-index staging pool is ever shared with a
    /// non-worker thread.
    ///
    /// @param slotCount Number of worker slots plus the reserved slot
    ///                  (`logicalCores + 1`).
    void initGenScratchArenas(unsigned slotCount);

}
