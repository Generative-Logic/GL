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

#include <cstdint>
#include <string>

namespace gl {

    struct LbMemory;

    /// @brief End-of-burst per-structure memory-size measurement — the memory
    ///        counterpart of the RT section table.
    ///
    /// @details
    /// RT answers "where does the time go" by attributing wall-clock to labelled
    /// scopes. This namespace answers "where does the memory go" by attributing
    /// bytes to statified structures. The two share a purpose, an output folder
    /// (`.rt/`), a compile-time gate convention, and the discipline that neither
    /// is ever read by the prover.
    ///
    /// **Three families are measured**, because they answer different questions
    /// and have different denominators:
    ///
    /// - **Per-LB deload-enrolled containers.** Every container
    ///   `LbMemory::visitContainers` reaches, keyed by its `ContainerTag`: the
    ///   ~60 direct `LbMemory` tags, the four `HashMemory` bands at 51 / 151 /
    ///   251 / 351, both `ColdMail` instances at 455 / 505, the changed-classes
    ///   buffer at 655, the name caches at 705, and the deloadable mail-out at
    ///   755. The figure is LOGICAL content bytes — `sizeof(value_type) *
    ///   size()`, exactly what the deload directory records — not the padded
    ///   page footprint.
    /// - **Derived indexes.** The `PagedHashIndex` slot arrays inside every cold
    ///   map, and `ReverseArgsIndex`. These are rebuilt on reload and never
    ///   deloaded (I-117 / I-154), so the deload image understates them; they
    ///   are real RAM all the same.
    /// - **Process-wide singletons.** The four pool footprints
    ///   (`blocksInUse() * blockBytes()` for Main / Persistent / Mail / Lb) and
    ///   the two scratch registries. These are PHYSICAL block bytes, a coarser
    ///   and strictly larger denominator than the logical container figure —
    ///   every LB with any content pins a whole block (G-52). They are reported
    ///   in their own section for exactly that reason, never mixed into the
    ///   per-structure percentages.
    ///
    /// **Why the sample lands where it does.** An LB's containers can only be
    /// read while the LB is resident: a deloaded cold map reads as empty and its
    /// residency assert fires (I-111, cookbook pitfall 12). The end of
    /// `performElemPhase3` is the last point at which the burst's LB is still
    /// claimed and loaded, so that is where a worker samples its own LB into its
    /// own slot row. The rows are folded at the end-of-iteration barrier, where
    /// the grid is quiesced and one coherent instant exists.
    ///
    /// @invariant Nothing here is a proof input. No prover branch, deload
    ///            decision, or steward verdict reads a counter (Rule 16, I-106).
    /// @invariant Heap-free on the sampling path: fixed static tables only, so a
    ///            gated-on build adds no allocation to a burst (Rule 28).
    /// @see `docs/agentic_swdd/_meta/memory_measurement.md`, `rt_tracker.hpp`.
    namespace mem_tracker {

        /// @brief The process-wide singleton rows, reported apart from the
        ///        per-structure table.
        ///
        /// @details
        /// These are physical block bytes rather than logical content bytes, so
        /// they share no denominator with the container tags. `Count` is the
        /// array bound and never a live slot.
        enum class GlobalSlot : int {
            PoolMain = 0,        ///< `staticMemory()` — deloadable per-LB store.
            PoolPersistent = 1,  ///< `persistentMemory()` — `intToBeProved`.
            PoolMail = 2,        ///< `mailMemory()` — `MailLog` + `mailIn`.
            PoolLb = 3,          ///< `lbMemory()` — the `LbStore` shell slab.
            ScratchString = 4,   ///< `scratchArenas()` peak across slots.
            ScratchGen = 5,      ///< `genScratchArenas()` peak across slots.
            Count = 6            ///< Array bound, not a slot.
        };

        /// @brief Clear the per-worker accumulator rows for a fresh iteration.
        ///
        /// @details
        /// Called from the single-threaded head of a `proveKernel` iteration,
        /// before any worker samples. Zeroes every slot row so the iteration's
        /// fold counts each active LB exactly once; LBs idle across the
        /// iteration contribute nothing, which is the intended reading (they are
        /// deloaded and therefore unpollable — see the namespace details).
        ///
        /// @invariant Single-threaded. Calling it while workers are running
        ///            would race their `addLbSample` writes.
        void resetIteration();

        /// @brief Fold one resident LB's per-container byte sizes into a
        ///        worker's own accumulator row.
        ///
        /// @details
        /// Walks `lb.visitContainers` and adds `sizeof(value_type) * size()` for
        /// each visited container under its `ContainerTag`. A container's FIRST
        /// facet also carries `indexBytes()`, so the derived hash index is
        /// attributed once, to the tag naming its container. Whatever the arena
        /// physically holds beyond content and index is slack — whole pinned
        /// blocks and the unused tail of every touched page — and lands in the
        /// slot's slack counter, which belongs to no tag. Writes only `slot`'s
        /// row, so concurrent workers never touch the same memory and no atomics
        /// are needed.
        ///
        /// @param lb   The logic block's statified aggregate. MUST be resident;
        ///             a deloaded aggregate trips the containers' own residency
        ///             asserts.
        /// @param slot The caller's worker slot (`coreId`), or the reserved last
        ///             slot on a single-threaded seam. Asserts when at or past
        ///             `MEM_MAX_SLOTS`.
        ///
        /// @invariant Read-only with respect to the aggregate: every accessor
        ///            used is a `const` size query.
        void addLbSample(const LbMemory& lb, unsigned slot);

        /// @brief Fold the worker rows, add the process-wide singletons, and
        ///        keep the sample if it is a new high-water.
        ///
        /// @details
        /// Sums the per-slot rows into one per-tag vector, reads the four pool
        /// footprints and the two scratch registries into the `GlobalSlot` row,
        /// and compares the grand total (containers plus derived indexes; the
        /// pool rows are context, not addends) against the running peak. On a
        /// new peak the whole vector is copied into the peak snapshot, so the
        /// reported table describes one real instant rather than a collage of
        /// per-structure maxima taken at different times.
        ///
        /// @invariant Single-threaded — called at the end-of-iteration barrier
        ///            with the grid quiesced.
        void commitIterationSample();

        /// @brief Write the peak snapshot as a percentage table.
        ///
        /// @details
        /// Rows are one per structure (facets of the same container folded
        /// together — the five `encodedMap` columns are one `encodedMap` row),
        /// sorted by descending bytes, each carrying its share of the peak grand
        /// total. Below the table come the process-wide pool and scratch
        /// figures, and a header naming the peak iteration and the sample count.
        /// Asserts on any I/O failure rather than returning a sentinel
        /// (Rule 19).
        ///
        /// @param path Destination file, conventionally
        ///             `.rt/_memory_<anchor>.log`.
        void dumpMemAggregate(const std::string& path);

        /// @brief Write the peak snapshot as a self-contained HTML tree.
        ///
        /// @details
        /// The same numbers as `dumpMemAggregate`, rendered as a collapsible
        /// three-level hierarchy — instance, container, facet — so a structure's
        /// cost can be read at whichever level the question is asked at. Each
        /// row carries content bytes, the container's derived-index bytes, their
        /// total, the share of the peak and the share of container content, plus
        /// a bar. No external stylesheet, script or font: the file opens
        /// offline. Asserts on any I/O failure (Rule 19).
        ///
        /// @param path Destination file, conventionally
        ///             `.rt/_memory_<anchor>.html`.
        void dumpMemHtml(const std::string& path);

        /// @brief Number of iteration samples committed so far.
        ///
        /// @details
        /// Exposed for the unit tests and for the dump header; a zero count
        /// means the gate was on but no iteration ever reached the barrier.
        ///
        /// @return Committed sample count since process start.
        int64_t sampleCount();

        /// @brief Grand total of the peak snapshot, in bytes.
        ///
        /// @details
        /// The denominator of every percentage in the dumped table: container
        /// content bytes plus derived-index bytes at the peak iteration. Zero
        /// before the first commit.
        ///
        /// @return Peak attributed bytes.
        int64_t peakTotalBytes();

        /// @brief Discard every accumulated counter and snapshot.
        ///
        /// @details
        /// Restores the process-start state so a unit test can run an
        /// independent scenario. Not called by the pipeline.
        void resetAllForTest();

    } // namespace mem_tracker

} // namespace gl
