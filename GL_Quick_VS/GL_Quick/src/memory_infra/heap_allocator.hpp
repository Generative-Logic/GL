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

#include <cassert>
#include <cstddef>

#ifdef USE_MIMALLOC
#include <mimalloc.h>
#endif

// Process-wide configuration of the heap allocator
// (D-427).

namespace gl {

    /// @brief Size in KiB of every mimalloc arena the process reserves: 2 GiB,
    ///        64 arena blocks of 32 MiB, exactly one 64-bit bitmap field.
    inline constexpr long kHeapArenaReserveKiB = 2L * 1024L * 1024L;

    /// @brief Make every mimalloc arena reserved from now on fill whole
    ///        bitmap fields, so blocks above 64 MiB are served from arenas and
    ///        returned to the operating system when freed.
    ///
    /// @details
    /// mimalloc 2.2.3 reserves 1 GiB arenas by default: 32 blocks of 32 MiB in
    /// a 64-bit bitmap field whose upper 32 bits it marks as permanently in
    /// use. Its claim of more than two consecutive blocks counts the free bits
    /// from the top of the field, finds none in such an arena and fails, so
    /// every block above 64 MiB (the Phase 2 host projection's large columns)
    /// falls back to a direct operating-system allocation after reserving one
    /// more unused arena. That fallback records the allocation with size zero,
    /// and the free path skips the release of a zero-size range: the memory
    /// stays committed for the rest of the process and is never reused
    /// (G-78). Arenas of 64 blocks have no
    /// permanently marked bits, so a fresh arena accepts any request up to its
    /// size and the fallback is never taken; freed arena blocks are purged by
    /// the collects the prover already runs.
    ///
    /// Whether a freed block is returned at the next collect or kept for reuse
    /// depends on the mimalloc version: 3.5.1 on macOS keeps the blocks of up
    /// to 112 MiB committed and reuses them for the next allocations, so the
    /// unit test checks that a second round commits no more than the first.
    ///
    /// Call once at process start, before the first large allocation. Arenas
    /// reserved earlier keep their size; requests they cannot place reserve a
    /// new, conforming arena. Without mimalloc the call does nothing.
    ///
    /// @return Nothing.
    /// @invariant mimalloc reports `kHeapArenaReserveKiB` KiB as its arena
    ///            reservation size after the call.
    /// @see kHeapArenaReserveKiB
    inline void configureHeapAllocator() {
#ifdef USE_MIMALLOC
        mi_option_set(mi_option_arena_reserve, kHeapArenaReserveKiB);
        assert(mi_option_get_size(mi_option_arena_reserve)
                   == static_cast<std::size_t>(kHeapArenaReserveKiB) * 1024u
               && "configureHeapAllocator: mimalloc ignored the arena size");
#endif
    }

} // namespace gl
