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

#include <cstdint>

// Operating-system address-space primitives behind the static pools
// (D-418): a pool reserves its whole range once and
// commits blocks before use. Returned whole blocks can be decommitted while
// retaining their reservation and recommitted at the same address on reuse.

namespace gl {

    /// @brief Reserve `bytes` of contiguous address space without committing
    ///        any of it.
    ///
    /// @details
    /// The range is inaccessible until `commitAddressSpace` commits a part of
    /// it; reserving costs no commit charge and no physical memory. On Windows
    /// this is `VirtualAlloc(MEM_RESERVE)` (64 KiB-aligned base); elsewhere an
    /// anonymous `PROT_NONE` mapping with `MAP_NORESERVE` (page-aligned base).
    /// Both bases are multiples of 4096, which the CUDA host registration of
    /// pool blocks requires. Failure is an assert: the static pools are sized
    /// at program start and have no fallback (Rule 19).
    ///
    /// @param bytes Size of the range; positive.
    /// @return Base of the reserved range, aligned to at least 4096 bytes.
    /// @see commitAddressSpace, releaseAddressSpace
    char* reserveAddressSpace(int64_t bytes);

    /// @brief Commit a sub-range of a reservation for reading and writing.
    ///
    /// @details
    /// The range is widened to whole operating-system pages; committing a page
    /// that is already committed leaves its content unchanged, so blocks
    /// smaller than a page may share one. Freshly committed pages read as zero
    /// and take physical memory only when first written. A commit the system
    /// cannot back (commit limit reached) is an assert.
    ///
    /// @param address First byte to commit; inside a live reservation.
    /// @param bytes   Number of bytes to commit; positive, inside the same
    ///                reservation.
    /// @see reserveAddressSpace
    void commitAddressSpace(char* address, int64_t bytes);

    /// @brief Decommit complete operating-system pages while retaining their addresses.
    ///
    /// @details
    /// Only unowned, unregistered pages inside a live reservation may be passed.
    /// The exact range becomes inaccessible and loses its contents; a subsequent
    /// commit makes zero-filled pages accessible at the same virtual addresses.
    /// Windows uses MEM_DECOMMIT; POSIX replaces this reserved subrange with an
    /// anonymous PROT_NONE mapping. No address outside the range is changed.
    ///
    /// @param address Page-aligned first byte inside the caller's reservation.
    /// @param bytes Positive whole-page length entirely owned by the caller.
    /// @return Nothing.
    /// @invariant Partial pages and operating-system failures assert.
    /// @see reserveAddressSpace, commitAddressSpace
    void decommitAddressSpace(char* address, int64_t bytes);

    /// @brief Return a whole reservation, committed parts included, to the
    ///        operating system.
    ///
    /// @details
    /// `address` and `bytes` must be exactly a `reserveAddressSpace` result
    /// and its size. Failure is an assert.
    ///
    /// @param address Base returned by `reserveAddressSpace`.
    /// @param bytes   The size passed to `reserveAddressSpace`.
    /// @see reserveAddressSpace
    void releaseAddressSpace(char* address, int64_t bytes);

} // namespace gl
